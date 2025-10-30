from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .video_backbone_lstm import FrameBackboneLSTM
from .video_torch import ArkoudiHead


class TabularEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int, hidden: Tuple[int, ...] = (256, 128), dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, emb_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VideoEncoderWrapper(nn.Module):
    def __init__(self, backbone_model: Optional[nn.Module] = None, **backbone_kwargs):
        super().__init__()
        # Note: `backbone_model` avoids clashing with FrameBackboneLSTM's `backbone` kwarg
        if backbone_model is not None:
            self.backbone = backbone_model
        else:
            self.backbone = FrameBackboneLSTM(**backbone_kwargs)

    @torch.no_grad()
    def output_dim(self) -> int:
        dummy = torch.zeros(1, 1, 3, 224, 224)
        _, z = self.backbone(dummy)
        return int(z.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, z = self.backbone(x)
        return z


class DeterministicMMVAE(nn.Module):
    def __init__(
        self,
        tab_in_dim: int,
        vid_backbone: Optional[nn.Module] = None,
        tab_emb_dim: int = 128,
        shared_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.0,
        video_kwargs: Optional[dict] = None,
        classifier_arkoudi: bool = True,
        fuse_dropout: float = 0.0,
    ):
        super().__init__()
        self.tab_enc = TabularEncoder(tab_in_dim, tab_emb_dim, dropout=dropout)
        self.vid_enc = VideoEncoderWrapper(backbone_model=vid_backbone, **(video_kwargs or {}))

        vid_emb_dim = self.vid_enc.output_dim()
        fuse_in = tab_emb_dim + vid_emb_dim
        fuse_hidden = max(shared_dim * 2, fuse_in // 2 + 1)
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, fuse_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=fuse_dropout) if fuse_dropout and fuse_dropout > 0 else nn.Identity(),
            nn.Linear(fuse_hidden, shared_dim),
        )

        self.dec_tab = nn.Sequential(
            nn.Linear(shared_dim, max(tab_emb_dim, shared_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(max(tab_emb_dim, shared_dim), tab_emb_dim),
        )
        self.dec_vid = nn.Sequential(
            nn.Linear(shared_dim, max(vid_emb_dim, shared_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(max(vid_emb_dim, shared_dim), vid_emb_dim),
        )

        if classifier_arkoudi:
            self.classifier = ArkoudiHead(shared_dim, num_classes, normalize=True)
        else:
            self.classifier = nn.Linear(shared_dim, num_classes)

    def encode_modalities(self, x_tab: torch.Tensor, x_vid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_tab = self.tab_enc(x_tab)
        z_vid = self.vid_enc(x_vid)
        return z_tab, z_vid

    def fuse_modalities(self, z_tab: torch.Tensor, z_vid: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_tab, z_vid], dim=-1)
        return self.fuse(z)

    def decode_modalities(self, z_shared: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rec_tab = self.dec_tab(z_shared)
        rec_vid = self.dec_vid(z_shared)
        return rec_tab, rec_vid

    def forward(self, x_tab: torch.Tensor, x_vid: torch.Tensor):
        z_tab, z_vid = self.encode_modalities(x_tab, x_vid)
        z_shared = self.fuse_modalities(z_tab, z_vid)
        rec_tab, rec_vid = self.decode_modalities(z_shared)
        logits = self.classifier(z_shared)
        return {
            "z_tab": z_tab,
            "z_vid": z_vid,
            "z": z_shared,
            "rec_tab": rec_tab,
            "rec_vid": rec_vid,
            "logits": logits,
        }

    def loss(
        self,
        out: dict,
        y: Optional[torch.Tensor] = None,
        w_rec_tab: float = 1.0,
        w_rec_vid: float = 1.0,
        w_cls: float = 1.0,
        label_smoothing: float = 0.0,
    ) -> Tuple[torch.Tensor, dict]:
        l_rec_tab = F.mse_loss(out["rec_tab"], out["z_tab"])
        l_rec_vid = F.mse_loss(out["rec_vid"], out["z_vid"])
        l_cls = torch.tensor(0.0, device=out["z"].device)
        if y is not None:
            l_cls = F.cross_entropy(out["logits"], y, label_smoothing=float(label_smoothing))
        total = w_rec_tab * l_rec_tab + w_rec_vid * l_rec_vid + w_cls * l_cls
        return total, {"rec_tab": l_rec_tab.item(), "rec_vid": l_rec_vid.item(), "cls": l_cls.item()}


class VariationalMMVAE(DeterministicMMVAE):
    def __init__(
        self,
        tab_in_dim: int,
        vid_backbone: Optional[nn.Module] = None,
        tab_emb_dim: int = 128,
        shared_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.0,
        video_kwargs: Optional[dict] = None,
        classifier_arkoudi: bool = True,
        kl_anneal_start: float = 0.0,
        kl_anneal_end: float = 1.0,
        kl_anneal_steps: int = 1000,
        fuse_dropout: float = 0.0,
    ):
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
        )
        enc_out = self.fuse[0].in_features  # concat size
        self.q_mu = nn.Linear(enc_out, shared_dim)
        self.q_logvar = nn.Linear(enc_out, shared_dim)
        self.fuse = None
        self._kl_anneal_start = float(kl_anneal_start)
        self._kl_anneal_end = float(kl_anneal_end)
        self._kl_anneal_steps = int(kl_anneal_steps)

    def fuse_modalities(self, z_tab: torch.Tensor, z_vid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.cat([z_tab, z_vid], dim=-1)
        mu = self.q_mu(h)
        logvar = self.q_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def forward(self, x_tab: torch.Tensor, x_vid: torch.Tensor):
        z_tab, z_vid = self.encode_modalities(x_tab, x_vid)
        z, mu, logvar = self.fuse_modalities(z_tab, z_vid)
        rec_tab, rec_vid = self.decode_modalities(z)
        logits = self.classifier(z)
        return {
            "z_tab": z_tab,
            "z_vid": z_vid,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "rec_tab": rec_tab,
            "rec_vid": rec_vid,
            "logits": logits,
        }

    def _kl_weight(self, step: int) -> float:
        if self._kl_anneal_steps <= 0:
            return self._kl_anneal_end
        t = max(0, min(step, self._kl_anneal_steps)) / float(self._kl_anneal_steps)
        return (1 - t) * self._kl_anneal_start + t * self._kl_anneal_end

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
    ) -> Tuple[torch.Tensor, dict]:
        l_rec_tab = F.mse_loss(out["rec_tab"], out["z_tab"])
        l_rec_vid = F.mse_loss(out["rec_vid"], out["z_vid"])
        l_cls = torch.tensor(0.0, device=out["z"].device)
        if y is not None:
            l_cls = F.cross_entropy(out["logits"], y, label_smoothing=float(label_smoothing))
        mu, logvar = out["mu"], out["logvar"]
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_w = self._kl_weight(step)
        total = w_rec_tab * l_rec_tab + w_rec_vid * l_rec_vid + w_cls * l_cls + w_kl * kl_w * kl
        return total, {"rec_tab": l_rec_tab.item(), "rec_vid": l_rec_vid.item(), "cls": l_cls.item(), "kl": kl.item(), "kl_w": kl_w}
