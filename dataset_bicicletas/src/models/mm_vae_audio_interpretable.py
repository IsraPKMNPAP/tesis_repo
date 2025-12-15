from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mm_vae_audio import DeterministicMMVAEAudio
from .mm_vae import _safe_normalize


def _proto_orthogonality_loss(head: nn.Module) -> torch.Tensor:
    """Promueve que los embeddings de clase sean ortogonales (interpretabilidad)."""
    device = None
    if hasattr(head, "class_emb"):
        w = head.class_emb
        device = w.device
    else:
        try:
            device = next(head.parameters()).device
        except StopIteration:
            device = None
        return torch.tensor(0.0, device=device)
    w = F.normalize(w, p=2, dim=1)
    gram = w @ w.t()
    eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
    return ((gram - eye) ** 2).mean()


def _margin_loss(z: torch.Tensor, y: Optional[torch.Tensor], mode: str = "mse") -> torch.Tensor:
    """Fuerza que la dimensión k responda a la clase k (inspirado en Arkoudi et al.)."""
    if y is None:
        return torch.tensor(0.0, device=z.device)
    num_dim = z.size(1)
    y_safe = y.clamp(min=0, max=num_dim - 1)
    if mode == "hinge":
        z_true = z.gather(1, y_safe.view(-1, 1))
        mask = torch.ones_like(z, dtype=torch.bool)
        mask.scatter_(1, y_safe.view(-1, 1), False)
        max_neg = z.masked_fill(~mask, float("-inf")).max(dim=1, keepdim=True).values
        margin = torch.clamp(1.0 - (z_true - max_neg), min=0.0)
        return margin.mean()
    # Default: MSE contra one-hot
    target = F.one_hot(y_safe, num_classes=num_dim).float()
    z_norm = F.normalize(z, p=2, dim=-1)
    return F.mse_loss(z_norm, target)


class InterpretableMMVAEAudioDeterministic(DeterministicMMVAEAudio):
    """Extiende el VAE determinista para forzar un embedding interpretable (dim ~ etiquetas)."""

    def __init__(
        self,
        tab_in_dim: int,
        tab_emb_dim: int,
        audio_emb_dim: int,
        shared_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        video_kwargs: Optional[dict] = None,
        classifier_arkoudi: bool = True,
        fuse_dropout: float = 0.0,
        proj_dim: int = 128,
        contrastive_temp: float = 0.07,
        modality_dropout_p: float = 0.0,
        fusion_type: str = "early",
        late_alpha: float = 0.5,
        audio_encoder: Optional[nn.Module] = None,
        audio_encoder_type: str = "simple",
        audio_kwargs: Optional[dict] = None,
    ):
        self.interpretable_dim = int(shared_dim)
        super().__init__(
            tab_in_dim=tab_in_dim,
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
            audio_emb_dim=audio_emb_dim,
            audio_encoder=audio_encoder,
            audio_encoder_type=audio_encoder_type,
            audio_kwargs=audio_kwargs,
        )
        # Proyección fija para audio (evita crear capas nuevas en cada forward)
        self.proj_aud = nn.Linear(self.audio_emb_dim, self.proj_tab.out_features)

    def forward(self, x_tab: torch.Tensor, x_vid: torch.Tensor, x_aud: Optional[torch.Tensor] = None):
        z_tab, z_vid = self.encode_modalities(x_tab, x_vid)
        z_aud = self.encode_audio(x_aud)
        if z_aud is None:
            z_aud = torch.zeros(z_tab.size(0), self.audio_emb_dim, device=z_tab.device, dtype=z_tab.dtype)
        p_tab = _safe_normalize(self.proj_tab(z_tab), dim=-1)
        p_vid = _safe_normalize(self.proj_vid(z_vid), dim=-1)
        p_aud = _safe_normalize(self.proj_aud(z_aud), dim=-1)
        z_shared = self.fuse_modalities(z_tab, z_vid, z_aud)
        rec_tab, rec_vid = self.decode_modalities(z_shared)
        logits_tab = self.cls_tab(z_tab)
        logits_vid = self.cls_vid(z_vid)
        logits_aud = self.cls_aud(z_aud) if z_aud is not None else None

        if self.fusion_type == "late":
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
                weights.append(0.33)
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
        w_l1_z: float = 0.0,
        w_proto_ortho: float = 0.0,
        w_margin: float = 0.0,
        margin_type: str = "mse",
    ):
        total, logs = super().loss(
            out=out,
            y=y,
            w_rec_tab=w_rec_tab,
            w_rec_vid=w_rec_vid,
            w_cls=w_cls,
            label_smoothing=label_smoothing,
            w_align=w_align,
            w_contrastive=w_contrastive,
            w_aux_tab=w_aux_tab,
            w_aux_vid=w_aux_vid,
            w_aux_aud=w_aux_aud,
            class_weights=class_weights,
        )

        l1_z = out["z"].abs().mean()
        proto_ortho = _proto_orthogonality_loss(self.classifier)
        margin = _margin_loss(out["z"], y, mode=margin_type)

        total = total + float(w_l1_z) * l1_z + float(w_proto_ortho) * proto_ortho + float(w_margin) * margin
        logs.update(
            {
                "l1_z": float(l1_z.item()),
                "proto_ortho": float(proto_ortho.item()),
                "margin": float(margin.item()),
            }
        )
        return total, logs


class InterpretableMMVAEAudioVariational(InterpretableMMVAEAudioDeterministic):
    """Versión variacional con embedding interpretable + regularizadores."""

    def __init__(
        self,
        tab_in_dim: int,
        tab_emb_dim: int,
        audio_emb_dim: int,
        shared_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        video_kwargs: Optional[dict] = None,
        classifier_arkoudi: bool = True,
        kl_anneal_start: float = 0.0,
        kl_anneal_end: float = 1.0,
        kl_anneal_steps: int = 1000,
        fuse_dropout: float = 0.0,
        proj_dim: int = 128,
        contrastive_temp: float = 0.07,
        modality_dropout_p: float = 0.0,
        fusion_type: str = "early",
        late_alpha: float = 0.5,
        audio_encoder: Optional[nn.Module] = None,
        audio_encoder_type: str = "simple",
        audio_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            tab_in_dim=tab_in_dim,
            tab_emb_dim=tab_emb_dim,
            audio_emb_dim=audio_emb_dim,
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
            audio_encoder=audio_encoder,
            audio_encoder_type=audio_encoder_type,
            audio_kwargs=audio_kwargs,
        )
        self.tab_emb_dim = tab_emb_dim
        self.vid_emb_dim = self.vid_enc.output_dim()
        self._kl_anneal_start = float(kl_anneal_start)
        self._kl_anneal_end = float(kl_anneal_end)
        self._kl_anneal_steps = int(kl_anneal_steps)

        fuse_in = self.tab_emb_dim + self.vid_emb_dim + self.audio_emb_dim
        self.q_mu = nn.Linear(fuse_in, shared_dim)
        self.q_logvar = nn.Linear(fuse_in, shared_dim)

    def _kl_weight(self, step: int) -> float:
        if self._kl_anneal_steps <= 0:
            return self._kl_anneal_end
        t = max(0, min(step, self._kl_anneal_steps)) / float(self._kl_anneal_steps)
        return (1 - t) * self._kl_anneal_start + t * self._kl_anneal_end

    def fuse_modalities(self, z_tab: torch.Tensor, z_vid: torch.Tensor, z_aud: Optional[torch.Tensor]) -> tuple:
        z_tab_d, z_vid_d = self._apply_modality_dropout(z_tab, z_vid)
        parts = [z_tab_d, z_vid_d]
        if z_aud is not None:
            parts.append(z_aud)
        h = torch.cat(parts, dim=-1)
        mu = self.q_mu(h)
        logvar = self.q_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def forward(self, x_tab: torch.Tensor, x_vid: torch.Tensor, x_aud: Optional[torch.Tensor] = None):
        z_tab, z_vid = self.encode_modalities(x_tab, x_vid)
        z_aud = self.encode_audio(x_aud)
        if z_aud is None:
            z_aud = torch.zeros(z_tab.size(0), self.audio_emb_dim, device=z_tab.device, dtype=z_tab.dtype)
        p_tab = _safe_normalize(self.proj_tab(z_tab), dim=-1)
        p_vid = _safe_normalize(self.proj_vid(z_vid), dim=-1)
        p_aud = _safe_normalize(self.proj_aud(z_aud), dim=-1)
        z_shared, mu, logvar = self.fuse_modalities(z_tab, z_vid, z_aud)
        rec_tab, rec_vid = self.decode_modalities(z_shared)
        logits_tab = self.cls_tab(z_tab)
        logits_vid = self.cls_vid(z_vid)
        logits_aud = self.cls_aud(z_aud) if z_aud is not None else None

        if self.fusion_type == "late":
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
                weights.append(0.33)
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
            "mu": mu,
            "logvar": logvar,
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
        w_kl: float = 1.0,
        step: int = 0,
        label_smoothing: float = 0.0,
        w_align: float = 0.0,
        w_contrastive: float = 0.0,
        w_aux_tab: float = 0.0,
        w_aux_vid: float = 0.0,
        w_aux_aud: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        w_l1_z: float = 0.0,
        w_proto_ortho: float = 0.0,
        w_margin: float = 0.0,
        margin_type: str = "mse",
    ):
        total, logs = super().loss(
            out=out,
            y=y,
            w_rec_tab=w_rec_tab,
            w_rec_vid=w_rec_vid,
            w_cls=w_cls,
            label_smoothing=label_smoothing,
            w_align=w_align,
            w_contrastive=w_contrastive,
            w_aux_tab=w_aux_tab,
            w_aux_vid=w_aux_vid,
            w_aux_aud=w_aux_aud,
            class_weights=class_weights,
            w_l1_z=w_l1_z,
            w_proto_ortho=w_proto_ortho,
            w_margin=w_margin,
            margin_type=margin_type,
        )
        mu, logvar = out["mu"], out["logvar"]
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_w = self._kl_weight(step)
        total = total + float(w_kl) * float(kl_w) * kl
        logs.update({"kl": float(kl.item()), "kl_w": float(kl_w)})
        return total, logs
