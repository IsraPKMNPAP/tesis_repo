from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mm_vae import TabularEncoder, VideoEncoderWrapper
from .video_torch import ArkoudiHead
from .audio_encoders import SimpleAudioEncoder


class CrossModalAttentionBlock(nn.Module):
    """Transformer-style cross-attention block over modality tokens."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ModalityGate(nn.Module):
    """Small gate to weight each modality embedding."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class CrossAttentiveMMVAEAudio(nn.Module):
    """
    Variational multimodal VAE with cross-attention and learned modality gates.
    Modalities: tabular, video, audio. Focused on predictive performance.
    """

    def __init__(
        self,
        tab_in_dim: int,
        tab_emb_dim: int = 128,
        vid_backbone: Optional[nn.Module] = None,
        audio_encoder: Optional[nn.Module] = None,
        audio_emb_dim: int = 128,
        fusion_dim: int = 128,
        video_kwargs: Optional[dict] = None,
        latent_dim: int = 64,
        num_heads: int = 4,
        attn_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        num_classes: int = 3,
        classifier_arkoudi: bool = True,
        proj_dim: int = 128,
        contrastive_temp: float = 0.07,
        modality_dropout_p: float = 0.0,
        kl_anneal_start: float = 0.0,
        kl_anneal_end: float = 1.0,
        kl_anneal_steps: int = 1000,
        fuse_dropout: float = 0.1,
    ):
        super().__init__()
        self.tab_enc = TabularEncoder(tab_in_dim, tab_emb_dim, dropout=dropout)
        self.vid_enc = VideoEncoderWrapper(backbone_model=vid_backbone, **(video_kwargs or {}))
        self.audio_enc = audio_encoder or SimpleAudioEncoder(emb_dim=audio_emb_dim)

        vid_emb_dim = self.vid_enc.output_dim()
        self.tab_proj = nn.Linear(tab_emb_dim, fusion_dim)
        self.vid_proj = nn.Linear(vid_emb_dim, fusion_dim)
        self.aud_proj = nn.Linear(audio_emb_dim, fusion_dim)

        self.gate_tab = ModalityGate(tab_emb_dim, hidden=max(32, tab_emb_dim // 2))
        self.gate_vid = ModalityGate(vid_emb_dim, hidden=max(32, vid_emb_dim // 2))
        self.gate_aud = ModalityGate(audio_emb_dim, hidden=max(32, audio_emb_dim // 2))

        self.fusion_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        nn.init.trunc_normal_(self.fusion_token, std=0.02)

        self.attn_blocks = nn.ModuleList(
            [
                CrossModalAttentionBlock(
                    dim=fusion_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(attn_layers)
            ]
        )

        self.norm_fuse = nn.LayerNorm(fusion_dim)
        fuse_hidden = max(fusion_dim * 2, latent_dim * 2)
        self.q_mu = nn.Sequential(nn.Linear(fusion_dim, fuse_hidden), nn.GELU(), nn.Dropout(fuse_dropout), nn.Linear(fuse_hidden, latent_dim))
        self.q_logvar = nn.Sequential(
            nn.Linear(fusion_dim, fuse_hidden), nn.GELU(), nn.Dropout(fuse_dropout), nn.Linear(fuse_hidden, latent_dim)
        )

        self.dec_tab = nn.Sequential(
            nn.Linear(latent_dim, max(latent_dim, tab_emb_dim)),
            nn.GELU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(max(latent_dim, tab_emb_dim), tab_emb_dim),
        )
        self.dec_vid = nn.Sequential(
            nn.Linear(latent_dim, max(latent_dim, vid_emb_dim)),
            nn.GELU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(max(latent_dim, vid_emb_dim), vid_emb_dim),
        )
        self.dec_aud = nn.Sequential(
            nn.Linear(latent_dim, max(latent_dim, audio_emb_dim)),
            nn.GELU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(max(latent_dim, audio_emb_dim), audio_emb_dim),
        )

        if classifier_arkoudi:
            self.classifier = ArkoudiHead(latent_dim, num_classes, normalize=True)
            self.cls_tab = ArkoudiHead(tab_emb_dim, num_classes, normalize=True)
            self.cls_vid = ArkoudiHead(vid_emb_dim, num_classes, normalize=True)
            self.cls_aud = ArkoudiHead(audio_emb_dim, num_classes, normalize=True)
        else:
            self.classifier = nn.Linear(latent_dim, num_classes)
            self.cls_tab = nn.Linear(tab_emb_dim, num_classes)
            self.cls_vid = nn.Linear(vid_emb_dim, num_classes)
            self.cls_aud = nn.Linear(audio_emb_dim, num_classes)

        self.proj_tab = nn.Linear(tab_emb_dim, proj_dim)
        self.proj_vid = nn.Linear(vid_emb_dim, proj_dim)
        self.proj_aud = nn.Linear(audio_emb_dim, proj_dim)
        self.contrastive_temp = float(contrastive_temp)
        self.modality_dropout_p = float(modality_dropout_p)

        self._kl_anneal_start = float(kl_anneal_start)
        self._kl_anneal_end = float(kl_anneal_end)
        self._kl_anneal_steps = int(kl_anneal_steps)

    def _maybe_modality_dropout(self, tokens: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.modality_dropout_p <= 0 or not self.training or len(tokens) <= 1:
            return tokens
        keep_mask: List[torch.Tensor] = [torch.ones_like(tokens[0][..., :1])]  # keep fusion token
        modality_masks = []
        for t in tokens[1:]:
            keep = (torch.rand(t.size(0), 1, 1, device=t.device) > self.modality_dropout_p).float()
            modality_masks.append(keep)
        stacked = torch.cat(modality_masks, dim=1)
        none_keep = stacked.sum(dim=1, keepdim=True) == 0
        if none_keep.any():
            modality_masks[0][none_keep] = 1.0
        keep_mask.extend(modality_masks)
        return [t * m for t, m in zip(tokens, keep_mask)]

    def _kl_weight(self, step: int) -> float:
        if self._kl_anneal_steps <= 0:
            return self._kl_anneal_end
        t = max(0, min(step, self._kl_anneal_steps)) / float(self._kl_anneal_steps)
        return (1 - t) * self._kl_anneal_start + t * self._kl_anneal_end

    def _contrastive_loss(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        B = p1.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=p1.device)
        logits = p1 @ p2.t() / max(1e-6, float(self.contrastive_temp))
        labels = torch.arange(B, device=p1.device)
        loss12 = F.cross_entropy(logits, labels)
        loss21 = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss12 + loss21)

    def encode(self, x_tab: torch.Tensor, x_vid: torch.Tensor, x_aud: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        z_tab = self.tab_enc(x_tab)
        z_vid = self.vid_enc(x_vid)
        z_aud = self.audio_enc(x_aud) if x_aud is not None else None
        return z_tab, z_vid, z_aud

    def fuse_modalities(self, z_tab: torch.Tensor, z_vid: torch.Tensor, z_aud: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        B = z_tab.size(0)
        tokens = []

        gate_tab = self.gate_tab(z_tab)
        gate_vid = self.gate_vid(z_vid)
        tab_tok = self.tab_proj(z_tab) * gate_tab
        vid_tok = self.vid_proj(z_vid) * gate_vid
        tokens.append(self.fusion_token.expand(B, -1, -1))  # fusion token at index 0
        tokens.append(tab_tok.unsqueeze(1))
        tokens.append(vid_tok.unsqueeze(1))

        gate_aud = None
        if z_aud is not None:
            gate_aud = self.gate_aud(z_aud)
            aud_tok = self.aud_proj(z_aud) * gate_aud
            tokens.append(aud_tok.unsqueeze(1))

        tokens = self._maybe_modality_dropout(tokens)
        tokens = torch.cat(tokens, dim=1)  # [B, n_tokens, fusion_dim]

        for blk in self.attn_blocks:
            tokens = blk(tokens)

        tokens = self.norm_fuse(tokens)
        fused = tokens[:, 0]
        mu = self.q_mu(fused)
        logvar = self.q_logvar(fused)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        gate_info = {"gate_tab": gate_tab, "gate_vid": gate_vid, "gate_aud": gate_aud}
        return z, mu, logvar, gate_info

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec_tab = self.dec_tab(z)
        rec_vid = self.dec_vid(z)
        rec_aud = self.dec_aud(z)
        return rec_tab, rec_vid, rec_aud

    def forward(self, x_tab: torch.Tensor, x_vid: torch.Tensor, x_aud: Optional[torch.Tensor] = None):
        z_tab, z_vid, z_aud = self.encode(x_tab, x_vid, x_aud)
        z, mu, logvar, gates = self.fuse_modalities(z_tab, z_vid, z_aud)
        rec_tab, rec_vid, rec_aud = self.decode(z)

        logits = self.classifier(z)
        logits_tab = self.cls_tab(z_tab)
        logits_vid = self.cls_vid(z_vid)
        logits_aud = self.cls_aud(z_aud) if z_aud is not None else None

        p_tab = F.normalize(self.proj_tab(z_tab), p=2, dim=-1)
        p_vid = F.normalize(self.proj_vid(z_vid), p=2, dim=-1)
        p_aud = None
        if z_aud is not None:
            p_aud = F.normalize(self.proj_aud(z_aud), p=2, dim=-1)

        return {
            "z_tab": z_tab,
            "z_vid": z_vid,
            "z_aud": z_aud,
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
            "p_tab": p_tab,
            "p_vid": p_vid,
            "p_aud": p_aud,
            "gates": gates,
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
        w_contrastive: float = 0.0,
        w_aux_tab: float = 0.0,
        w_aux_vid: float = 0.0,
        w_aux_aud: float = 0.0,
        w_gate_reg: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        l_rec_tab = F.mse_loss(out["rec_tab"], out["z_tab"])
        l_rec_vid = F.mse_loss(out["rec_vid"], out["z_vid"])
        l_rec_aud = torch.tensor(0.0, device=out["z"].device)
        if out.get("z_aud") is not None and out["z_aud"] is not None:
            l_rec_aud = F.mse_loss(out["rec_aud"], out["z_aud"])

        l_cls = torch.tensor(0.0, device=out["z"].device)
        if y is not None:
            l_cls = F.cross_entropy(out["logits"], y, label_smoothing=float(label_smoothing), weight=class_weights)

        mu, logvar = out["mu"], out["logvar"]
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_w = self._kl_weight(step)

        l_con = torch.tensor(0.0, device=out["z"].device)
        if float(w_contrastive) > 0:
            l_con = self._contrastive_loss(out["p_tab"], out["p_vid"])
            if out.get("p_aud") is not None:
                l_con = l_con + 0.5 * (self._contrastive_loss(out["p_tab"], out["p_aud"]) + self._contrastive_loss(out["p_vid"], out["p_aud"]))

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

        l_gate = torch.tensor(0.0, device=out["z"].device)
        if float(w_gate_reg) > 0:
            gates = out.get("gates", {})
            terms = []
            for g in [gates.get("gate_tab"), gates.get("gate_vid"), gates.get("gate_aud")]:
                if g is not None:
                    # Encourage confident gates (close to 0 or 1)
                    terms.append((g * (1 - g)).mean())
            if terms:
                l_gate = sum(terms) / len(terms)

        total = (
            w_rec_tab * l_rec_tab
            + w_rec_vid * l_rec_vid
            + w_rec_aud * l_rec_aud
            + w_cls * l_cls
            + w_kl * kl_w * kl
            + float(w_contrastive) * l_con
            + float(w_aux_tab) * l_aux_tab
            + float(w_aux_vid) * l_aux_vid
            + float(w_aux_aud) * l_aux_aud
            + float(w_gate_reg) * l_gate
        )
        return total, {
            "rec_tab": l_rec_tab.item(),
            "rec_vid": l_rec_vid.item(),
            "rec_aud": l_rec_aud.item() if isinstance(l_rec_aud, torch.Tensor) else 0.0,
            "cls": l_cls.item(),
            "kl": kl.item(),
            "kl_w": kl_w,
            "con": l_con.item() if isinstance(l_con, torch.Tensor) else 0.0,
            "aux_tab": l_aux_tab.item() if isinstance(l_aux_tab, torch.Tensor) else 0.0,
            "aux_vid": l_aux_vid.item() if isinstance(l_aux_vid, torch.Tensor) else 0.0,
            "aux_aud": l_aux_aud.item() if isinstance(l_aux_aud, torch.Tensor) else 0.0,
            "gate": l_gate.item() if isinstance(l_gate, torch.Tensor) else 0.0,
        }
