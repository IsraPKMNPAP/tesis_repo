from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _normalize(x: torch.Tensor, mean, std):
    mean = x.new_tensor(mean)[None, :, None, None]
    std = x.new_tensor(std)[None, :, None, None]
    return (x - mean) / std


def _unfreeze_last_blocks(root: nn.Module, attr_paths: list[str], last_n: int):
    if last_n <= 0:
        return
    for path in attr_paths:
        try:
            # resolve dotted path to submodule list
            parts = path.split('.')
            m = root
            for p in parts:
                m = getattr(m, p)
            if isinstance(m, (nn.ModuleList, list, tuple)):
                blocks = list(m)
                for blk in blocks[-last_n:]:
                    for p in blk.parameters():
                        p.requires_grad = True
                return
        except Exception:
            continue


class ViTFrameEncoder(nn.Module):
    """ViT backbone encoder returning per-image embeddings.

    Uses torchvision or timm if available. Falls back to a simple projection if neither found.
    """

    def __init__(self, model_name: str = "vit_b_16", emb_dim: int = 768, trainable: bool = False, target_size: int = 224, unfreeze_last_n: int = 0):
        super().__init__()
        self.target_size = target_size
        self.normalize_stats = (IMAGENET_MEAN, IMAGENET_STD)
        backbone = None
        proj = None
        try:
            import torchvision.models as tvm

            if model_name == "vit_b_16":
                backbone = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.DEFAULT)
            elif model_name == "vit_b_32":
                backbone = tvm.vit_b_32(weights=tvm.ViT_B_32_Weights.DEFAULT)
            else:
                backbone = getattr(tvm, model_name)(weights="DEFAULT")
            emb_dim = backbone.heads.head.in_features
            backbone.heads.head = nn.Identity()
        except Exception:
            try:
                import timm

                backbone = timm.create_model(model_name, pretrained=True)
                if hasattr(backbone, "reset_classifier"):
                    backbone.reset_classifier(0)
                # Try to infer embed dim
                if hasattr(backbone, "num_features"):
                    emb_dim = backbone.num_features
            except Exception:
                # Fallback tiny conv + pool
                proj = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(64, emb_dim),
                )

        self.backbone = backbone
        self.proj = proj

        # Freeze if not trainable
        if self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if trainable and unfreeze_last_n > 0:
                # Try common block paths for torchvision and timm
                _unfreeze_last_blocks(self.backbone, [
                    'encoder.layers',     # torchvision ViT
                    'blocks',             # timm ViT
                ], last_n=unfreeze_last_n)

        self.emb_dim = emb_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W] in [0,1] or [0,255]
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        if x.shape[-1] != self.target_size or x.shape[-2] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        x = _normalize(x, *self.normalize_stats)
        if self.backbone is not None:
            e = self.backbone(x)
        else:
            e = self.proj(x)
        return e


class ClipFrameEncoder(nn.Module):
    """CLIP image encoder via open-clip. Returns per-image embeddings before CLIP logit scale.

    If open_clip is unavailable, falls back to a small projection.
    """

    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "openai", trainable: bool = False, target_size: int = 224, emb_dim: int = 512, unfreeze_last_n: int = 0):
        super().__init__()
        self.target_size = target_size
        self.normalize_stats = (CLIP_MEAN, CLIP_STD)
        self.emb_dim = emb_dim
        self.is_open_clip = False
        try:
            import open_clip

            model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            # Extract visual encoder module
            self.visual = model.visual
            # Infer embedding dim
            if hasattr(self.visual, "output_dim"):
                self.emb_dim = getattr(self.visual, "output_dim")
            self.is_open_clip = True
            for p in self.visual.parameters():
                p.requires_grad = False
            if trainable and unfreeze_last_n > 0:
                # open-clip transformer blocks path
                try:
                    _unfreeze_last_blocks(self.visual, [
                        'transformer.resblocks',
                        'blocks',
                    ], last_n=unfreeze_last_n)
                except Exception:
                    pass
        except Exception:
            # Fallback projection
            self.visual = None
            self.fallback = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, self.emb_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        if x.shape[-1] != self.target_size or x.shape[-2] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        x = _normalize(x, *self.normalize_stats)
        if self.is_open_clip:
            # open-clip expects [N, C, H, W] float normalized
            e = self.visual(x)
            if isinstance(e, tuple):
                e = e[0]
        else:
            e = self.fallback(x)
        return e
