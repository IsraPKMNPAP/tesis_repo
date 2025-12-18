from __future__ import annotations

import torch
import torch.nn as nn


class EEGCNN(nn.Module):
    """
    CNN simple para EEG segments de forma [B, C, T].
    """

    def __init__(self, in_channels: int, hidden: int = 32, kernel_size: int = 7, dropout: float = 0.2):
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
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        h = self.conv(x)
        h = h.squeeze(-1)
        out = self.head(h)
        return out.squeeze(-1)
