from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torchaudio


class AudioCNNLogit(nn.Module):
    """Baseline CNN que convierte audio raw -> logits multinomiales."""

    def __init__(
        self,
        sample_rate: int,
        num_classes: int,
        n_mels: int = 64,
        cnn_channels: Sequence[int] = (32, 64, 128),
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
        in_ch = 1
        for out_ch in cnn_channels:
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_ch, num_classes)

    def _wave_to_spec(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.dim() == 2:
            waveforms = waveforms.unsqueeze(1)
        spec = self.melspec(waveforms)
        spec = self.to_db(spec).clamp(min=-80.0, max=80.0)
        if spec.dim() == 4 and spec.size(1) > 1:
            spec = spec.mean(dim=1, keepdim=True)
        return spec

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        spec = self._wave_to_spec(waveforms)
        feats = self.backbone(spec)
        feats = self.pool(feats).flatten(1)
        return self.classifier(feats)

    def extract_repr(self, waveforms: torch.Tensor) -> torch.Tensor:
        spec = self._wave_to_spec(waveforms)
        feats = self.backbone(spec)
        feats = self.pool(feats).flatten(1)
        return feats

    @property
    def repr_dim(self) -> int:
        # Dimensión previa a la capa de clasificación
        return self.classifier.in_features
