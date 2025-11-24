from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAudioEncoder(nn.Module):
    """Pequeño encoder 1D para formas de onda.

    Espera entrada [B, 1, T] (mono) y devuelve un embedding L2-normalizado.
    """

    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(128, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        h = self.net(x)
        h = h.view(h.size(0), -1)
        z = self.proj(h)
        return F.normalize(z, p=2, dim=-1)

