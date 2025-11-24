from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torchaudio


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu1(self.bn1(out))
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(self.bn2(out))
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class AudioTCNLogit(nn.Module):
    """TCN sobre log-Mel para clasificar acciones."""

    def __init__(
        self,
        sample_rate: int,
        num_classes: int,
        n_mels: int = 64,
        tcn_channels: Sequence[int] = (64, 128, 256),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=256,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

        layers = []
        in_ch = n_mels
        for i, out_ch in enumerate(tcn_channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_ch, num_classes)

    def _wave_to_feature(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.dim() == 2:
            waveforms = waveforms.unsqueeze(1)
        spec = self.melspec(waveforms)
        spec = self.to_db(spec).clamp(min=-80.0, max=80.0)
        # spec shape: (B, C, n_mels, time) cuando hay canales; colapsar a mono
        if spec.dim() == 4:
            spec = spec.mean(dim=1)
        elif spec.dim() != 3:
            raise ValueError(f"Forma inesperada del espectrograma: {spec.shape}")
        # resultado: (B, n_mels, time)
        return spec

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        feats = self._wave_to_feature(waveforms)
        out = self.tcn(feats)
        out = self.pool(out).flatten(1)
        return self.classifier(out)
