from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mm_vae import DeterministicMMVAE, VariationalMMVAE, _safe_normalize
from .audio_encoders import SimpleAudioEncoder


class DeterministicMMVAEAudio(DeterministicMMVAE):
    """Extiende el VAE determinista para soportar audio opcional (tab, video, audio)."""

    def __init__(
        self,
        tab_in_dim: int,
        vid_backbone: Optional[nn.Module] = None,
        audio_encoder: Optional[nn.Module] = None,
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
        audio_emb_dim: int = 128,
    ):
        self.audio_emb_dim = audio_emb_dim
        # Llamar al init padre (tab+video)
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
        # Crear encoder de audio y head auxiliar una vez inicializado el nn.Module padre
        self.audio_enc = audio_encoder or SimpleAudioEncoder(emb_dim=self.audio_emb_dim)
        self.proj_aud = nn.Linear(self.audio_emb_dim, self.proj_tab.out_features)
        # Ajustar classifier para soportar audio en late fusion
        if classifier_arkoudi:
            self.cls_aud = nn.Linear(self.audio_emb_dim, num_classes, bias=False)
        else:
            self.cls_aud = nn.Linear(self.audio_emb_dim, num_classes)

    def encode_audio(self, x_aud: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x_aud is None:
            return None
        return self.audio_enc(x_aud)

    def fuse_modalities(self, z_tab: torch.Tensor, z_vid: torch.Tensor, z_aud: Optional[torch.Tensor]) -> torch.Tensor:
        # Aplicar modality dropout solo a tab/vid; audio se incluye si existe
        z_tab_d, z_vid_d = self._apply_modality_dropout(z_tab, z_vid)
        parts = [z_tab_d, z_vid_d]
        if z_aud is not None:
            parts.append(z_aud)
        z = torch.cat(parts, dim=-1)
        return self.fuse(z)

    def forward(self, x_tab: torch.Tensor, x_vid: torch.Tensor, x_aud: Optional[torch.Tensor] = None):
        z_tab, z_vid = self.encode_modalities(x_tab, x_vid)
        z_aud = self.encode_audio(x_aud)
        p_tab = F.normalize(self.proj_tab(z_tab), p=2, dim=-1)
        p_vid = F.normalize(self.proj_vid(z_vid), p=2, dim=-1)
        p_aud = None
        if z_aud is not None:
            p_aud = _safe_normalize(self.proj_aud(z_aud), dim=-1)
        z_shared = self.fuse_modalities(z_tab, z_vid, z_aud)
        rec_tab, rec_vid = self.decode_modalities(z_shared)
        logits_tab = self.cls_tab(z_tab)
        logits_vid = self.cls_vid(z_vid)
        logits_aud = self.cls_aud(z_aud) if z_aud is not None else None

        if self.fusion_type == "late":
            # Promedio con pesos: alpha distribuye solo entre las modalidades presentes
            logits_list = []
            weights = []
            if logits_tab is not None:
                logits_list.append(logits_tab)
                weights.append(self.late_alpha / 2 if logits_aud is not None else self.late_alpha)
            if logits_vid is not None:
                logits_list.append(logits_vid)
                weights.append(1 - self.late_alpha if logits_aud is None else (1 - self.late_alpha) / 2)
            if logits_aud is not None:
                logits_list.append(logits_aud)
                weights.append(0.33)  # simple peso neutro si hay audio
            weights = [w / sum(weights) for w in weights]
            logits = sum(w * l for w, l in zip(weights, logits_list))
        else:
            logits = self.classifier(z_shared)

        return {
            "z_tab": z_tab,
            "z_vid": z_vid,
            "z_aud": z_aud,
            "p_tab": p_tab,
            "p_vid": p_vid,
            "p_aud": p_aud,
            "z": z_shared,
            "rec_tab": rec_tab,
            "rec_vid": rec_vid,
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
        w_cls: float = 1.0,
        label_smoothing: float = 0.0,
        w_align: float = 0.0,
        w_contrastive: float = 0.0,
        w_aux_tab: float = 0.0,
        w_aux_vid: float = 0.0,
        w_aux_aud: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        l_rec_tab = F.mse_loss(out["rec_tab"], out["z_tab"])
        l_rec_vid = F.mse_loss(out["rec_vid"], out["z_vid"])
        l_cls = torch.tensor(0.0, device=out["z"].device)
        if y is not None:
            l_cls = F.cross_entropy(out["logits"], y, label_smoothing=float(label_smoothing), weight=class_weights)

        l_align = torch.tensor(0.0, device=out["z"].device)
        if float(w_align) > 0:
            cos = F.cosine_similarity(out["p_tab"], out["p_vid"], dim=-1)
            l_align = 1.0 - cos.mean()

        l_con = torch.tensor(0.0, device=out["z"].device)
        if float(w_contrastive) > 0:
            l_con = self._contrastive_loss(out["p_tab"], out["p_vid"], self.contrastive_temp)

        # Aux CE per modality
        l_aux_tab = torch.tensor(0.0, device=out["z"].device)
        l_aux_vid = torch.tensor(0.0, device=out["z"].device)
        l_aux_aud = torch.tensor(0.0, device=out["z"].device)
        if y is not None:
            if out.get("logits_tab") is not None and float(w_aux_tab) > 0:
                l_aux_tab = F.cross_entropy(out["logits_tab"], y, label_smoothing=float(label_smoothing), weight=class_weights)
            if out.get("logits_vid") is not None and float(w_aux_vid) > 0:
                l_aux_vid = F.cross_entropy(out["logits_vid"], y, label_smoothing=float(label_smoothing), weight=class_weights)
            if out.get("logits_aud") is not None and out["logits_aud"] is not None and float(w_aux_aud) > 0:
                l_aux_aud = F.cross_entropy(out["logits_aud"], y, label_smoothing=float(label_smoothing), weight=class_weights)

        total = (
            w_rec_tab * l_rec_tab
            + w_rec_vid * l_rec_vid
            + w_cls * l_cls
            + float(w_align) * l_align
            + float(w_contrastive) * l_con
            + float(w_aux_tab) * l_aux_tab
            + float(w_aux_vid) * l_aux_vid
            + float(w_aux_aud) * l_aux_aud
        )
        return total, {
            "rec_tab": l_rec_tab.item(),
            "rec_vid": l_rec_vid.item(),
            "cls": l_cls.item(),
            "align": l_align.item() if isinstance(l_align, torch.Tensor) else 0.0,
            "con": l_con.item() if isinstance(l_con, torch.Tensor) else 0.0,
            "aux_tab": l_aux_tab.item() if isinstance(l_aux_tab, torch.Tensor) else 0.0,
            "aux_vid": l_aux_vid.item() if isinstance(l_aux_vid, torch.Tensor) else 0.0,
            "aux_aud": l_aux_aud.item() if isinstance(l_aux_aud, torch.Tensor) else 0.0,
        }


class VariationalMMVAEAudio(DeterministicMMVAEAudio, VariationalMMVAE):
    """Versión variacional con audio; hereda lógica de pérdidas/kl del padre variacional."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss(
        self,
        out: dict,
        y: Optional[torch.Tensor] = None,
        w_rec_tab: float = 1.0,
        w_rec_vid: float = 1.0,
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
        l_rec_tab = F.mse_loss(out["rec_tab"], out["z_tab"])
        l_rec_vid = F.mse_loss(out["rec_vid"], out["z_vid"])
        l_cls = torch.tensor(0.0, device=out["z"].device)
        if y is not None:
            l_cls = F.cross_entropy(out["logits"], y, label_smoothing=float(label_smoothing), weight=class_weights)
        mu, logvar = out["mu"], out["logvar"]
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_w = self._kl_weight(step)

        l_align = torch.tensor(0.0, device=out["z"].device)
        if float(w_align) > 0:
            cos = F.cosine_similarity(out["p_tab"], out["p_vid"], dim=-1)
            l_align = 1.0 - cos.mean()
        l_con = torch.tensor(0.0, device=out["z"].device)
        if float(w_contrastive) > 0:
            l_con = self._contrastive_loss(out["p_tab"], out["p_vid"], self.contrastive_temp)

        l_aux_tab = torch.tensor(0.0, device=out["z"].device)
        l_aux_vid = torch.tensor(0.0, device=out["z"].device)
        l_aux_aud = torch.tensor(0.0, device=out["z"].device)
        if y is not None:
            if out.get("logits_tab") is not None and float(w_aux_tab) > 0:
                l_aux_tab = F.cross_entropy(out["logits_tab"], y, label_smoothing=float(label_smoothing), weight=class_weights)
            if out.get("logits_vid") is not None and float(w_aux_vid) > 0:
                l_aux_vid = F.cross_entropy(out["logits_vid"], y, label_smoothing=float(label_smoothing), weight=class_weights)
            if out.get("logits_aud") is not None and out["logits_aud"] is not None and float(w_aux_aud) > 0:
                l_aux_aud = F.cross_entropy(out["logits_aud"], y, label_smoothing=float(label_smoothing), weight=class_weights)

        total = (
            w_rec_tab * l_rec_tab
            + w_rec_vid * l_rec_vid
            + w_cls * l_cls
            + w_kl * kl_w * kl
            + float(w_align) * l_align
            + float(w_contrastive) * l_con
            + float(w_aux_tab) * l_aux_tab
            + float(w_aux_vid) * l_aux_vid
            + float(w_aux_aud) * l_aux_aud
        )
        return total, {
            "rec_tab": l_rec_tab.item(),
            "rec_vid": l_rec_vid.item(),
            "cls": l_cls.item(),
            "kl": kl.item(),
            "kl_w": kl_w,
            "align": l_align.item() if isinstance(l_align, torch.Tensor) else 0.0,
            "con": l_con.item() if isinstance(l_con, torch.Tensor) else 0.0,
            "aux_tab": l_aux_tab.item() if isinstance(l_aux_tab, torch.Tensor) else 0.0,
            "aux_vid": l_aux_vid.item() if isinstance(l_aux_vid, torch.Tensor) else 0.0,
            "aux_aud": l_aux_aud.item() if isinstance(l_aux_aud, torch.Tensor) else 0.0,
        }
