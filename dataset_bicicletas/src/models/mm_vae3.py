from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mm_vae import DeterministicMMVAE, _safe_normalize
from .audio_encoders import SimpleAudioEncoder
from .audio_cnn import AudioCNNLogit
from .audio_tcn import AudioTCNLogit
from .audio_wav2vec import AudioWav2VecLogit


def _build_decoder(in_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Module:
    hidden = max(in_dim * 2, out_dim * 2)
    layers = [
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
        nn.Linear(hidden, out_dim),
    ]
    return nn.Sequential(*layers)


class DeterministicMMVAE3(DeterministicMMVAE):
    """MMVAE 3-modal (tab + video + audio) con reconstruccion de embeddings compactos."""

    def __init__(
        self,
        tab_in_dim: int,
        vid_backbone: Optional[nn.Module] = None,
        audio_encoder: Optional[nn.Module] = None,
        audio_encoder_type: str = "simple",
        audio_kwargs: Optional[dict] = None,
        tab_emb_dim: int = 128,
        shared_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.0,
        video_kwargs: Optional[dict] = None,
        classifier_arkoudi: bool = True,
        fuse_dropout: float = 0.0,
        proj_dim: int = 128,
        contrastive_temp: float = 0.07,
        modality_dropout_p: float = 0.0,
        fusion_type: str = "early",
        late_alpha: float = 0.5,
        late_mode: str = "mix",
        audio_emb_dim: int = 128,
        rec_emb_dim: int = 128,
    ):
        self.audio_emb_dim = audio_emb_dim
        self.tab_emb_dim = tab_emb_dim
        self.late_mode = late_mode
        self.rec_emb_dim = int(rec_emb_dim)
        if audio_encoder is None:
            audio_encoder, enc_dim = self._build_audio_encoder(
                encoder_type=audio_encoder_type,
                audio_kwargs=audio_kwargs or {},
                default_emb_dim=audio_emb_dim,
                num_classes=num_classes,
            )
            self.audio_emb_dim = enc_dim
        super().__init__(
            tab_in_dim=tab_in_dim,
            vid_backbone=vid_backbone,
            tab_emb_dim=tab_emb_dim,
            shared_dim=shared_dim,
            num_classes=num_classes,
            dropout=dropout,
            video_kwargs=video_kwargs,
            classifier_arkoudi=classifier_arkoudi,
            fuse_dropout=fuse_dropout,
            proj_dim=proj_dim,
            contrastive_temp=contrastive_temp,
            modality_dropout_p=modality_dropout_p,
            fusion_type=fusion_type,
            late_alpha=late_alpha,
        )
        self.vid_emb_dim = self.vid_enc.output_dim()
        fuse_in = self.tab_emb_dim + self.vid_emb_dim + self.audio_emb_dim
        fuse_hidden = max(shared_dim * 2, fuse_in // 2 + 1)
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, fuse_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=fuse_dropout) if fuse_dropout and fuse_dropout > 0 else nn.Identity(),
            nn.Linear(fuse_hidden, shared_dim),
        )
        self.dec_tab = _build_decoder(shared_dim, self.tab_emb_dim, dropout=fuse_dropout)
        self.dec_vid = _build_decoder(shared_dim, self.rec_emb_dim, dropout=fuse_dropout)
        self.dec_aud = _build_decoder(shared_dim, self.rec_emb_dim, dropout=fuse_dropout)
        self.proj_rec_vid = nn.Linear(self.vid_emb_dim, self.rec_emb_dim)
        self.proj_rec_aud = nn.Linear(self.audio_emb_dim, self.rec_emb_dim)

        self.audio_enc = audio_encoder
        self.proj_aud = nn.Linear(self.audio_emb_dim, self.proj_tab.out_features)
        if classifier_arkoudi:
            self.cls_aud = nn.Linear(self.audio_emb_dim, num_classes, bias=False)
        else:
            self.cls_aud = nn.Linear(self.audio_emb_dim, num_classes)

    def encode_audio(self, x_aud: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x_aud is None or x_aud.numel() == 0:
            return None
        if x_aud.shape[-1] < 9:
            pad_len = 9 - x_aud.shape[-1]
            x_aud = torch.nn.functional.pad(x_aud, (0, pad_len))
        return self.audio_enc(x_aud)

    def _build_audio_encoder(
        self,
        encoder_type: str,
        audio_kwargs: dict,
        default_emb_dim: int,
        num_classes: int,
    ) -> Tuple[nn.Module, int]:
        et = (encoder_type or "simple").lower()
        if et == "simple":
            emb_dim = audio_kwargs.get("emb_dim", default_emb_dim)
            return SimpleAudioEncoder(emb_dim=emb_dim), emb_dim
        if et == "cnn":
            sample_rate = int(audio_kwargs.get("sample_rate", 16000))
            n_mels = int(audio_kwargs.get("n_mels", 64))
            channels = audio_kwargs.get("cnn_channels", (32, 64, 128))
            dropout = float(audio_kwargs.get("dropout", 0.2))
            model = AudioCNNLogit(sample_rate=sample_rate, num_classes=num_classes, n_mels=n_mels, cnn_channels=channels, dropout=dropout)
            emb_dim = model.repr_dim

            class CNNEncoder(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, x):
                    return self.m.extract_repr(x)

            return CNNEncoder(model), emb_dim
        if et == "tcn":
            sample_rate = int(audio_kwargs.get("sample_rate", 16000))
            n_mels = int(audio_kwargs.get("n_mels", 64))
            channels = audio_kwargs.get("tcn_channels", (64, 128, 256))
            kernel = int(audio_kwargs.get("kernel_size", 3))
            dropout = float(audio_kwargs.get("dropout", 0.2))
            model = AudioTCNLogit(sample_rate=sample_rate, num_classes=num_classes, n_mels=n_mels, tcn_channels=channels, kernel_size=kernel, dropout=dropout)
            emb_dim = model.repr_dim

            class TCNEncoder(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, x):
                    return self.m.extract_repr(x)

            return TCNEncoder(model), emb_dim
        if et == "wav2vec":
            bundle_name = audio_kwargs.get("bundle_name", "WAV2VEC2_BASE")
            trainable = bool(audio_kwargs.get("trainable", False))
            dropout = float(audio_kwargs.get("dropout", 0.1))
            model = AudioWav2VecLogit(bundle_name=bundle_name, num_classes=num_classes, trainable=trainable, dropout=dropout)
            emb_dim = model.repr_dim

            class W2VEncoder(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, x):
                    return self.m.extract_repr(x)

            return W2VEncoder(model), emb_dim
        raise ValueError(f"Tipo de encoder de audio no soportado: {encoder_type}")

    def fuse_modalities(self, z_tab: torch.Tensor, z_vid: torch.Tensor, z_aud: Optional[torch.Tensor]) -> torch.Tensor:
        z_tab_d, z_vid_d = self._apply_modality_dropout(z_tab, z_vid)
        parts = [z_tab_d, z_vid_d]
        if z_aud is not None:
            parts.append(z_aud)
        return self.fuse(torch.cat(parts, dim=-1))

    def forward(self, x_tab: torch.Tensor, x_vid: torch.Tensor, x_aud: Optional[torch.Tensor] = None):
        z_tab, z_vid = self.encode_modalities(x_tab, x_vid)
        z_aud = self.encode_audio(x_aud)
        if z_aud is None:
            z_aud = torch.zeros(z_tab.size(0), self.audio_emb_dim, device=z_tab.device, dtype=z_tab.dtype)
        p_tab = _safe_normalize(self.proj_tab(z_tab), dim=-1)
        p_vid = _safe_normalize(self.proj_vid(z_vid), dim=-1)
        p_aud = _safe_normalize(self.proj_aud(z_aud), dim=-1)
        z_shared = self.fuse_modalities(z_tab, z_vid, z_aud)
        rec_tab = self.dec_tab(z_shared)
        rec_vid = self.dec_vid(z_shared)
        rec_aud = self.dec_aud(z_shared)
        z_vid_rec = self.proj_rec_vid(z_vid)
        z_aud_rec = self.proj_rec_aud(z_aud)
        logits_tab = self.cls_tab(z_tab)
        logits_vid = self.cls_vid(z_vid)
        logits_aud = self.cls_aud(z_aud)

        if self.fusion_type == "late":
            if self.late_mode == "tab_only":
                logits = logits_tab
            elif self.late_mode == "vid_only":
                logits = logits_vid
            elif self.late_mode == "aud_only":
                logits = logits_aud
            else:
                weights = [self.late_alpha / 2, self.late_alpha / 2, 1 - self.late_alpha]
                total_w = sum(weights)
                weights = [w / total_w for w in weights]
                logits = weights[0] * logits_tab + weights[1] * logits_vid + weights[2] * logits_aud
        else:
            logits = self.classifier(z_shared)

        return {
            "z_tab": z_tab,
            "z_vid": z_vid,
            "z_aud": z_aud,
            "z_vid_rec": z_vid_rec,
            "z_aud_rec": z_aud_rec,
            "p_tab": p_tab,
            "p_vid": p_vid,
            "p_aud": p_aud,
            "z": z_shared,
            "rec_tab": rec_tab,
            "rec_vid": rec_vid,
            "rec_aud": rec_aud,
            "logits": logits,
            "logits_tab": logits_tab,
            "logits_vid": logits_vid,
            "logits_aud": logits_aud,
        }

    def loss(
        self,
        out: dict,
        y: Optional[torch.Tensor] = None,
        w_rec_tab: float = 1.0,
        w_rec_vid: float = 1.0,
        w_rec_aud: float = 1.0,
        w_cls: float = 1.0,
        label_smoothing: float = 0.0,
        w_align: float = 0.0,
        w_contrastive: float = 0.0,
        w_aux_tab: float = 0.0,
        w_aux_vid: float = 0.0,
        w_aux_aud: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        l_rec_tab = F.mse_loss(out["rec_tab"], out["z_tab"].detach())
        l_rec_vid = F.mse_loss(out["rec_vid"], out["z_vid_rec"].detach())
        l_rec_aud = F.mse_loss(out["rec_aud"], out["z_aud_rec"].detach())
        l_cls = torch.tensor(0.0, device=out["z"].device)
        if y is not None:
            l_cls = F.cross_entropy(out["logits"], y, label_smoothing=float(label_smoothing), weight=class_weights)

        l_align = torch.tensor(0.0, device=out["z"].device)
        if float(w_align) > 0:
            cos_tv = F.cosine_similarity(out["p_tab"], out["p_vid"], dim=-1)
            cos_ta = F.cosine_similarity(out["p_tab"], out["p_aud"], dim=-1)
            cos_va = F.cosine_similarity(out["p_vid"], out["p_aud"], dim=-1)
            l_align = 1.0 - (cos_tv + cos_ta + cos_va).mean() / 3.0

        l_con = torch.tensor(0.0, device=out["z"].device)
        if float(w_contrastive) > 0:
            c_tv = self._contrastive_loss(out["p_tab"], out["p_vid"], self.contrastive_temp)
            c_ta = self._contrastive_loss(out["p_tab"], out["p_aud"], self.contrastive_temp)
            c_va = self._contrastive_loss(out["p_vid"], out["p_aud"], self.contrastive_temp)
            l_con = (c_tv + c_ta + c_va) / 3.0

        l_aux_tab = torch.tensor(0.0, device=out["z"].device)
        l_aux_vid = torch.tensor(0.0, device=out["z"].device)
        l_aux_aud = torch.tensor(0.0, device=out["z"].device)
        if y is not None:
            if out.get("logits_tab") is not None and float(w_aux_tab) > 0:
                l_aux_tab = F.cross_entropy(out["logits_tab"], y, label_smoothing=float(label_smoothing), weight=class_weights)
            if out.get("logits_vid") is not None and float(w_aux_vid) > 0:
                l_aux_vid = F.cross_entropy(out["logits_vid"], y, label_smoothing=float(label_smoothing), weight=class_weights)
            if out.get("logits_aud") is not None and float(w_aux_aud) > 0:
                l_aux_aud = F.cross_entropy(out["logits_aud"], y, label_smoothing=float(label_smoothing), weight=class_weights)

        total = (
            w_rec_tab * l_rec_tab
            + w_rec_vid * l_rec_vid
            + w_rec_aud * l_rec_aud
            + w_cls * l_cls
            + float(w_align) * l_align
            + float(w_contrastive) * l_con
            + float(w_aux_tab) * l_aux_tab
            + float(w_aux_vid) * l_aux_vid
            + float(w_aux_aud) * l_aux_aud
        )
        total = torch.nan_to_num(total)
        return total, {
            "rec_tab": float(torch.nan_to_num(l_rec_tab).item()),
            "rec_vid": float(torch.nan_to_num(l_rec_vid).item()),
            "rec_aud": float(torch.nan_to_num(l_rec_aud).item()),
            "cls": float(torch.nan_to_num(l_cls).item()),
            "align": float(torch.nan_to_num(l_align).item()),
            "con": float(torch.nan_to_num(l_con).item()),
            "aux_tab": float(torch.nan_to_num(l_aux_tab).item()),
            "aux_vid": float(torch.nan_to_num(l_aux_vid).item()),
            "aux_aud": float(torch.nan_to_num(l_aux_aud).item()),
        }


class VariationalMMVAE3(DeterministicMMVAE3):
    """Version variacional con audio (mu/logvar + reparam)."""

    def __init__(
        self,
        tab_in_dim: int,
        vid_backbone: Optional[nn.Module] = None,
        audio_encoder: Optional[nn.Module] = None,
        audio_encoder_type: str = "simple",
        audio_kwargs: Optional[dict] = None,
        tab_emb_dim: int = 128,
        shared_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.0,
        video_kwargs: Optional[dict] = None,
        classifier_arkoudi: bool = True,
        fuse_dropout: float = 0.0,
        proj_dim: int = 128,
        contrastive_temp: float = 0.07,
        modality_dropout_p: float = 0.0,
        fusion_type: str = "early",
        late_alpha: float = 0.5,
        late_mode: str = "mix",
        audio_emb_dim: int = 128,
        rec_emb_dim: int = 128,
        kl_anneal_start: float = 0.0,
        kl_anneal_end: float = 1.0,
        kl_anneal_steps: int = 1000,
    ):
        super().__init__(
            tab_in_dim=tab_in_dim,
            vid_backbone=vid_backbone,
            audio_encoder=audio_encoder,
            audio_encoder_type=audio_encoder_type,
            audio_kwargs=audio_kwargs,
            tab_emb_dim=tab_emb_dim,
            shared_dim=shared_dim,
            num_classes=num_classes,
            dropout=dropout,
            video_kwargs=video_kwargs,
            classifier_arkoudi=classifier_arkoudi,
            fuse_dropout=fuse_dropout,
            proj_dim=proj_dim,
            contrastive_temp=contrastive_temp,
            modality_dropout_p=modality_dropout_p,
            fusion_type=fusion_type,
            late_alpha=late_alpha,
            late_mode=late_mode,
            audio_emb_dim=audio_emb_dim,
            rec_emb_dim=rec_emb_dim,
        )
        enc_out = self.fuse[0].in_features
        self.q_mu = nn.Linear(enc_out, shared_dim)
        self.q_logvar = nn.Linear(enc_out, shared_dim)
        self.fuse = None
        self._kl_anneal_start = float(kl_anneal_start)
        self._kl_anneal_end = float(kl_anneal_end)
        self._kl_anneal_steps = int(kl_anneal_steps)

    def fuse_modalities(self, z_tab: torch.Tensor, z_vid: torch.Tensor, z_aud: torch.Tensor):
        z_tab_d, z_vid_d = self._apply_modality_dropout(z_tab, z_vid)
        z_cat = torch.cat([z_tab_d, z_vid_d, z_aud], dim=-1)
        mu = self.q_mu(z_cat)
        logvar = self.q_logvar(z_cat)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def _kl_weight(self, step: int) -> float:
        if self._kl_anneal_steps <= 0:
            return self._kl_anneal_end
        t = max(0, min(step, self._kl_anneal_steps)) / float(self._kl_anneal_steps)
        return (1 - t) * self._kl_anneal_start + t * self._kl_anneal_end

    def forward(self, x_tab: torch.Tensor, x_vid: torch.Tensor, x_aud: Optional[torch.Tensor] = None):
        z_tab, z_vid = self.encode_modalities(x_tab, x_vid)
        z_aud = self.encode_audio(x_aud)
        if z_aud is None:
            z_aud = torch.zeros(z_tab.size(0), self.audio_emb_dim, device=z_tab.device, dtype=z_tab.dtype)
        p_tab = _safe_normalize(self.proj_tab(z_tab), dim=-1)
        p_vid = _safe_normalize(self.proj_vid(z_vid), dim=-1)
        p_aud = _safe_normalize(self.proj_aud(z_aud), dim=-1)
        z, mu, logvar = self.fuse_modalities(z_tab, z_vid, z_aud)
        rec_tab = self.dec_tab(z)
        rec_vid = self.dec_vid(z)
        rec_aud = self.dec_aud(z)
        z_vid_rec = self.proj_rec_vid(z_vid)
        z_aud_rec = self.proj_rec_aud(z_aud)
        logits_tab = self.cls_tab(z_tab)
        logits_vid = self.cls_vid(z_vid)
        logits_aud = self.cls_aud(z_aud)
        if self.fusion_type == "late":
            if self.late_mode == "tab_only":
                logits = logits_tab
            elif self.late_mode == "vid_only":
                logits = logits_vid
            elif self.late_mode == "aud_only":
                logits = logits_aud
            else:
                weights = [self.late_alpha / 2, self.late_alpha / 2, 1 - self.late_alpha]
                total_w = sum(weights)
                weights = [w / total_w for w in weights]
                logits = weights[0] * logits_tab + weights[1] * logits_vid + weights[2] * logits_aud
        else:
            logits = self.classifier(z)
        return {
            "z_tab": z_tab,
            "z_vid": z_vid,
            "z_aud": z_aud,
            "z_vid_rec": z_vid_rec,
            "z_aud_rec": z_aud_rec,
            "p_tab": p_tab,
            "p_vid": p_vid,
            "p_aud": p_aud,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "rec_tab": rec_tab,
            "rec_vid": rec_vid,
            "rec_aud": rec_aud,
            "logits": logits,
            "logits_tab": logits_tab,
            "logits_vid": logits_vid,
            "logits_aud": logits_aud,
        }

    def loss(
        self,
        out: dict,
        y: Optional[torch.Tensor] = None,
        w_rec_tab: float = 1.0,
        w_rec_vid: float = 1.0,
        w_rec_aud: float = 1.0,
        w_cls: float = 1.0,
        w_kl: float = 1.0,
        step: int = 0,
        label_smoothing: float = 0.0,
        w_align: float = 0.0,
        w_contrastive: float = 0.0,
        w_aux_tab: float = 0.0,
        w_aux_vid: float = 0.0,
        w_aux_aud: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        l_rec_tab = F.mse_loss(out["rec_tab"], out["z_tab"].detach())
        l_rec_vid = F.mse_loss(out["rec_vid"], out["z_vid_rec"].detach())
        l_rec_aud = F.mse_loss(out["rec_aud"], out["z_aud_rec"].detach())
        l_cls = torch.tensor(0.0, device=out["z"].device)
        if y is not None:
            l_cls = F.cross_entropy(out["logits"], y, label_smoothing=float(label_smoothing), weight=class_weights)
        mu, logvar = out["mu"], out["logvar"]
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_w = self._kl_weight(step)

        l_align = torch.tensor(0.0, device=out["z"].device)
        if float(w_align) > 0:
            cos_tv = F.cosine_similarity(out["p_tab"], out["p_vid"], dim=-1)
            cos_ta = F.cosine_similarity(out["p_tab"], out["p_aud"], dim=-1)
            cos_va = F.cosine_similarity(out["p_vid"], out["p_aud"], dim=-1)
            l_align = 1.0 - (cos_tv + cos_ta + cos_va).mean() / 3.0
        l_con = torch.tensor(0.0, device=out["z"].device)
        if float(w_contrastive) > 0:
            c_tv = self._contrastive_loss(out["p_tab"], out["p_vid"], self.contrastive_temp)
            c_ta = self._contrastive_loss(out["p_tab"], out["p_aud"], self.contrastive_temp)
            c_va = self._contrastive_loss(out["p_vid"], out["p_aud"], self.contrastive_temp)
            l_con = (c_tv + c_ta + c_va) / 3.0

        l_aux_tab = torch.tensor(0.0, device=out["z"].device)
        l_aux_vid = torch.tensor(0.0, device=out["z"].device)
        l_aux_aud = torch.tensor(0.0, device=out["z"].device)
        if y is not None:
            if out.get("logits_tab") is not None and float(w_aux_tab) > 0:
                l_aux_tab = F.cross_entropy(out["logits_tab"], y, label_smoothing=float(label_smoothing), weight=class_weights)
            if out.get("logits_vid") is not None and float(w_aux_vid) > 0:
                l_aux_vid = F.cross_entropy(out["logits_vid"], y, label_smoothing=float(label_smoothing), weight=class_weights)
            if out.get("logits_aud") is not None and float(w_aux_aud) > 0:
                l_aux_aud = F.cross_entropy(out["logits_aud"], y, label_smoothing=float(label_smoothing), weight=class_weights)

        total = (
            w_rec_tab * l_rec_tab
            + w_rec_vid * l_rec_vid
            + w_rec_aud * l_rec_aud
            + w_cls * l_cls
            + w_kl * kl_w * kl
            + float(w_align) * l_align
            + float(w_contrastive) * l_con
            + float(w_aux_tab) * l_aux_tab
            + float(w_aux_vid) * l_aux_vid
            + float(w_aux_aud) * l_aux_aud
        )
        total = torch.nan_to_num(total)
        return total, {
            "rec_tab": float(torch.nan_to_num(l_rec_tab).item()),
            "rec_vid": float(torch.nan_to_num(l_rec_vid).item()),
            "rec_aud": float(torch.nan_to_num(l_rec_aud).item()),
            "cls": float(torch.nan_to_num(l_cls).item()),
            "kl": float(torch.nan_to_num(kl).item()),
            "kl_w": kl_w,
            "align": float(torch.nan_to_num(l_align).item()),
            "con": float(torch.nan_to_num(l_con).item()),
            "aux_tab": float(torch.nan_to_num(l_aux_tab).item()),
            "aux_vid": float(torch.nan_to_num(l_aux_vid).item()),
            "aux_aud": float(torch.nan_to_num(l_aux_aud).item()),
        }
