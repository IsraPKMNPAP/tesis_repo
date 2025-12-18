from __future__ import annotations

import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128, 64), dropout: float = 0.2):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)])
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
