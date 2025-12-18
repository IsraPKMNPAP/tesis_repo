from __future__ import annotations

import torch
import torch.nn as nn


class TabImageFusion(nn.Module):
    """
    Fusión temprana: proyecta embedding de imagen y concatena con tabular.
    """

    def __init__(self, tab_dim: int, img_dim: int, img_proj: int = 128, hidden=(128, 64), dropout: float = 0.2):
        super().__init__()
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, img_proj),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        layers = []
        last = tab_dim + img_proj
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)])
            last = h
        layers.append(nn.Linear(last, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, tab: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        img_f = self.img_proj(img)
        x = torch.cat([tab, img_f], dim=-1)
        return self.mlp(x).squeeze(-1)
