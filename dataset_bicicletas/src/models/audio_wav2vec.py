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
        self.repr_dim = feat_dim

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # Espera (B, T); si viene con canal, promediar a mono
        if waveforms.dim() == 3:
            if waveforms.size(1) > 1:
                waveforms = waveforms.mean(dim=1)
            else:
                waveforms = waveforms.squeeze(1)
        elif waveforms.dim() == 2:
            pass
        else:
            raise ValueError(f"Forma inesperada para wav2vec: {waveforms.shape}")
        outputs = self.backbone.extract_features(waveforms)
        # extract_features devuelve (features, lengths) o lista de features; manejar ambos
        if isinstance(outputs, tuple):
            features, _ = outputs
        else:
            features = outputs
        if isinstance(features, list):
            features = features[-1]
        # features: (B, T, C)
        feats = features.transpose(1, 2)  # (B, C, T)
        pooled = self.pool(feats).squeeze(-1)
        return self.head(pooled)

    def extract_repr(self, waveforms: torch.Tensor) -> torch.Tensor:
        # Igual que forward pero devolviendo el embedding antes del head
        if waveforms.dim() == 3:
            if waveforms.size(1) > 1:
                waveforms = waveforms.mean(dim=1)
            else:
                waveforms = waveforms.squeeze(1)
        elif waveforms.dim() != 2:
            raise ValueError(f"Forma inesperada para wav2vec: {waveforms.shape}")
        outputs = self.backbone.extract_features(waveforms)
        if isinstance(outputs, tuple):
            features, _ = outputs
        else:
            features = outputs
        if isinstance(features, list):
            features = features[-1]
        feats = features.transpose(1, 2)
        pooled = self.pool(feats).squeeze(-1)
        return pooled
