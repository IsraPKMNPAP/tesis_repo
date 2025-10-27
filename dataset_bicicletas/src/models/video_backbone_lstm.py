from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_backbones import ViTFrameEncoder, ClipFrameEncoder
from .video_torch import ArkoudiHead


class FrameBackboneLSTM(nn.Module):
    """General frame encoder (CLIP/ViT) + LSTM over time + head.

    Input shapes supported: [B,T,C,H,W] or [T,C,H,W] or [B,C,H,W] (T=1).
    """

    def __init__(
        self,
        backbone: str = "vit",
        backbone_name: str = "vit_b_16",
        backbone_trainable: bool = False,
        backbone_unfreeze_last: int = 0,
        target_size: int = 224,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        bidirectional: bool = False,
        num_classes: int = 5,
        arkoudi: bool = True,
        arkoudi_normalize: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if backbone.lower() == "clip":
            self.frame_encoder = ClipFrameEncoder(model_name=backbone_name, trainable=backbone_trainable, target_size=target_size, unfreeze_last_n=backbone_unfreeze_last)
        else:
            self.frame_encoder = ViTFrameEncoder(model_name=backbone_name, trainable=backbone_trainable, target_size=target_size, unfreeze_last_n=backbone_unfreeze_last)
        enc_dim = getattr(self.frame_encoder, "emb_dim", 768)

        self.lstm = nn.LSTM(
            input_size=enc_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=(dropout if lstm_layers > 1 else 0.0),
        )
        head_in = lstm_hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        if arkoudi:
            self.head = ArkoudiHead(head_in, num_classes, normalize=arkoudi_normalize)
        else:
            self.head = nn.Linear(head_in, num_classes)

    def forward(self, x: torch.Tensor):
        # Normalize input into [B,T,C,H,W]
        if x.dim() == 4:
            if x.size(1) in (1, 3) and x.size(0) > 8:
                x = x.unsqueeze(0)
            else:
                x = x.unsqueeze(1)
        elif x.dim() == 3:
            x = x.unsqueeze(0).unsqueeze(1)

        B, T, C, H, W = x.shape
        xt = x.view(B * T, C, H, W)
        feat = self.frame_encoder(xt)  # [B*T, enc_dim]
        feat = feat.view(B, T, -1)
        out, _ = self.lstm(feat)
        z = out[:, -1, :]
        logits = self.head(self.dropout(z))
        return logits, z
