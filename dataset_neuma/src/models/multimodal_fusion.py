from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TabEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden=(128,), out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        if input_dim == 0:
            self.net = None
            self.out_dim = 0
        else:
            layers = []
            last = input_dim
            for h in hidden:
                layers.extend([nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)])
                last = h
            layers.append(nn.Linear(last, out_dim))
            self.net = nn.Sequential(*layers)
            self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            return torch.zeros((x.shape[0], 0), device=x.device, dtype=x.dtype)
        return self.net(x)


class EEGEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32, kernel_size: int = 7, out_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden * 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).squeeze(-1)
        return self.proj(h)


class FusionClassifier(nn.Module):
    """Deterministic early fusion."""

    def __init__(self, tab_dim: int, clip_dim: int, eeg_ch: int, tab_out: int = 64, eeg_out: int = 64, hidden=(128, 64), dropout: float = 0.2, img_proj: int = 0):
        super().__init__()
        self.tab_enc = TabEncoder(tab_dim, hidden=(128,), out_dim=tab_out, dropout=dropout)
        self.eeg_enc = EEGEncoder(eeg_ch, hidden=32, out_dim=eeg_out, dropout=dropout) if eeg_ch > 0 else None
        self.img_proj = None
        if img_proj and img_proj > 0:
            self.img_proj = nn.Sequential(nn.Linear(clip_dim, img_proj), nn.ReLU(), nn.Dropout(dropout))
            fused_in = self.tab_enc.out_dim + (eeg_out if self.eeg_enc is not None else 0) + img_proj
        else:
            fused_in = self.tab_enc.out_dim + (eeg_out if self.eeg_enc is not None else 0) + clip_dim

        layers = []
        last = fused_in
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)])
            last = h
        layers.append(nn.Linear(last, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, tab: torch.Tensor, clip: torch.Tensor, eeg: torch.Tensor) -> torch.Tensor:
        t = self.tab_enc(tab)
        e = self.eeg_enc(eeg) if self.eeg_enc is not None else torch.zeros((tab.shape[0], 0), device=tab.device, dtype=tab.dtype)
        c = self.img_proj(clip) if self.img_proj is not None else clip
        x = torch.cat([t, c, e], dim=-1)
        return self.head(x).squeeze(-1)


class FusionVAE(nn.Module):
    """Variational version: encoder->mu/logvar, reparam, decoder->logits."""

    def __init__(self, tab_dim: int, clip_dim: int, eeg_ch: int, tab_out: int = 64, eeg_out: int = 64, hidden=(256, 128), latent_dim: int = 64, dropout: float = 0.2, img_proj: int = 0):
        super().__init__()
        self.tab_enc = TabEncoder(tab_dim, hidden=(128,), out_dim=tab_out, dropout=dropout)
        self.eeg_enc = EEGEncoder(eeg_ch, hidden=32, out_dim=eeg_out, dropout=dropout) if eeg_ch > 0 else None
        self.img_proj = None
        if img_proj and img_proj > 0:
            self.img_proj = nn.Sequential(nn.Linear(clip_dim, img_proj), nn.ReLU(), nn.Dropout(dropout))
            enc_in = self.tab_enc.out_dim + (eeg_out if self.eeg_enc is not None else 0) + img_proj
        else:
            enc_in = self.tab_enc.out_dim + (eeg_out if self.eeg_enc is not None else 0) + clip_dim

        self.enc_mlp = nn.Sequential(
            nn.Linear(enc_in, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden[1], latent_dim)
        self.logvar = nn.Linear(hidden[1], latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], 1),
        )

    def encode(self, tab: torch.Tensor, clip: torch.Tensor, eeg: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        t = self.tab_enc(tab)
        e = self.eeg_enc(eeg) if self.eeg_enc is not None else torch.zeros((tab.shape[0], 0), device=tab.device, dtype=tab.dtype)
        c = self.img_proj(clip) if self.img_proj is not None else clip
        x = torch.cat([t, c, e], dim=-1)
        h = self.enc_mlp(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, tab: torch.Tensor, clip: torch.Tensor, eeg: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        mu, logvar = self.encode(tab, clip, eeg)
        z = self.reparam(mu, logvar)
        logits = self.dec(z).squeeze(-1)
        return logits, mu, logvar

    @staticmethod
    def kl_div(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
