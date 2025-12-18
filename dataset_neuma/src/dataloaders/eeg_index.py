from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def _crop_or_pad(arr: np.ndarray, target_len: int) -> np.ndarray:
    """arr shape [C, T]. Pads with zeros to the right if shorter; center-crops if longer."""
    c, t = arr.shape
    if t == target_len:
        return arr
    if t < target_len:
        out = np.zeros((c, target_len), dtype=arr.dtype)
        out[:, :t] = arr
        return out
    # t > target_len: center crop
    start = max(0, (t - target_len) // 2)
    end = start + target_len
    return arr[:, start:end]


class EEGIndexDataset(Dataset):
    def __init__(
        self,
        index_csv: Path,
        segment_len: int = 512,
        cache: bool = True,
    ) -> None:
        df = pd.read_csv(index_csv)
        df.columns = df.columns.str.lower()
        df = df.dropna(subset=["bought", "npy_path", "start", "end"])
        self.df = df.reset_index(drop=True)
        self.segment_len = segment_len
        self.cache_data: Dict[str, np.ndarray] = {} if cache else None

    def __len__(self) -> int:
        return len(self.df)

    def _load_array(self, path: str) -> np.ndarray:
        if self.cache_data is None:
            return np.load(path)
        if path not in self.cache_data:
            self.cache_data[path] = np.load(path)
        return self.cache_data[path]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        npy_path = row["npy_path"]
        data = self._load_array(npy_path)  # [C, T]
        start, end = int(row["start"]), int(row["end"])
        end = min(end, data.shape[1] - 1)
        segment = data[:, start : end + 1]
        segment = _crop_or_pad(segment, self.segment_len)
        x = torch.tensor(segment, dtype=torch.float32)
        y = torch.tensor(row["bought"], dtype=torch.float32)
        return x, y


def load_eeg_index(
    index_csv: Path,
    batch_size: int,
    segment_len: int = 512,
    cache: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    ds = EEGIndexDataset(index_csv=index_csv, segment_len=segment_len, cache=cache)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
