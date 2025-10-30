from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

from .video_windows import VideoWindowsDataset


@dataclass
class MultimodalSample:
    x_tab: torch.Tensor
    x_vid: torch.Tensor
    y: Optional[int]
    timestamp: Optional[str]
    window_id: Optional[int]
    participant: Optional[str]


class MultimodalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tab_columns: Sequence[str],
        X_tab_array: Optional[torch.Tensor] = None,
        path_col: str = "gpu_tensor_path",
        label_col: Optional[str] = None,
        timestamp_col: Optional[str] = "timestamp",
        window_id_col: Optional[str] = "window",
        prefer_df_label: bool = True,
        class_map: Optional[dict] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.tab_columns = list(tab_columns)
        self.X_tab_array = X_tab_array  # if provided, must align with df index
        self.label_col = label_col
        # Inner video dataset to reuse robust loading logic
        self.video_ds = VideoWindowsDataset(
            self.df,
            path_col=path_col,
            label_col=label_col,
            timestamp_col=timestamp_col,
            window_id_col=window_id_col,
            transform=None,
            prefer_df_label=prefer_df_label,
            class_map=class_map,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> MultimodalSample:
        v = self.video_ds[idx]
        row = self.df.iloc[idx]
        if self.X_tab_array is not None:
            x_tab = self.X_tab_array[idx]
        else:
            x_tab = torch.tensor(row[self.tab_columns].astype(float).values, dtype=torch.float32)
        x_vid = v.x
        y = v.y
        return MultimodalSample(
            x_tab=x_tab,
            x_vid=x_vid,
            y=y,
            timestamp=v.timestamp,
            window_id=v.window_id,
            participant=v.participant,
        )


def collate_multimodal(batch: List[MultimodalSample]):
    x_tabs = [b.x_tab for b in batch]
    x_vids = [b.x_vid for b in batch]
    ys = []
    ts, wids, parts = [], [], []

    def _coerce_label(v):
        if v is None:
            return -1
        try:
            if isinstance(v, torch.Tensor):
                return int(v.item())
            return int(v)
        except Exception:
            return -1

    for b in batch:
        ys.append(_coerce_label(b.y))
        ts.append(b.timestamp)
        wids.append(b.window_id)
        parts.append(b.participant)

    X_tab = torch.stack(x_tabs, dim=0)  # [B, D]

    # Video stacking: support [T,C,H,W] or [C,H,W] per item
    if x_vids[0].dim() in (3, 4):
        X_vid = torch.stack(x_vids, dim=0)
    else:
        raise ValueError(f"Dimensión de video no soportada: {x_vids[0].shape}")

    y = torch.tensor(ys, dtype=torch.long)

    class B:
        def __init__(self, x_tab, x_vid, y, timestamp, window_id, participant):
            self.x_tab = x_tab
            self.x_vid = x_vid
            self.y = y
            self.timestamp = timestamp
            self.window_id = window_id
            self.participant = participant

    return B(x_tab=X_tab, x_vid=X_vid, y=y, timestamp=ts, window_id=wids, participant=parts)
