from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torchaudio


def get_bundle(name: str):
    if not hasattr(torchaudio.pipelines, name):
        raise ValueError(f"Bundle de wav2vec no encontrado: {name}")
    return getattr(torchaudio.pipelines, name)


def _infer_feat_dim(model) -> int:
    if hasattr(model, "encoder") and hasattr(model.encoder, "transformer"):
        return getattr(model.encoder.transformer, "d_model", 768)
    return getattr(model, "encoder_embed_dim", 768)


class AudioWav2VecLogit(nn.Module):
    """Clasificador simple sobre embeddings de wav2vec 2.0."""

    def __init__(
        self,
        bundle_name: str,
        num_classes: int,
        trainable: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        bundle = get_bundle(bundle_name)
        self.sample_rate = bundle.sample_rate
        self.backbone = bundle.get_model()
        feat_dim = _infer_feat_dim(self.backbone)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
        if not trainable:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # waveforms: (B, 1, T) or (B, T)
        if waveforms.dim() == 2:
            waveforms = waveforms.unsqueeze(1)
        # wav2vec espera (B, T); quitar canal
        waveforms = waveforms.squeeze(1)
        features, _ = self.backbone.extract_features(waveforms)
        # features: (B, T, C)
        feats = features.transpose(1, 2)  # (B, C, T)
        pooled = self.pool(feats).squeeze(-1)
        return self.head(pooled)
