from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGAutoencoder(nn.Module):
    """
    Autoencoder simple 1D para EEG concatenado.
    - Encoder: Conv1d -> Conv1d -> GAP -> Linear -> emb_dim
    - Decoder: Linear -> ConvTranspose1d -> ConvTranspose1d para reconstruir la señal.
    """

    def __init__(self, in_channels: int, eeg_len: int, emb_dim: int = 64, hidden: int = 64, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.eeg_len = eeg_len
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden * 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc_enc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, emb_dim))

        self.fc_dec = nn.Sequential(nn.Linear(emb_dim, hidden * 2), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden * 2, hidden, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden, in_channels, kernel_size=kernel_size, padding=padding),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).squeeze(-1)
        z = self.fc_enc(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).unsqueeze(-1)  # [B, hidden*2, 1]
        # Expand temporal dimension to eeg_len
        h = h.expand(-1, -1, self.eeg_len)
        x_hat = self.decoder(h)
        return x_hat

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
