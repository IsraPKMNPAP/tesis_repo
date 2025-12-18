from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class EEGSegmentsDataset(Dataset):
    def __init__(
        self,
        npy_path: Path,
        segments_csv: Path,
        channels: List[int] = None,
    ) -> None:
        self.data = np.load(npy_path)  # shape [n_channels, n_samples]
        if channels:
            self.data = self.data[channels, :]
        import pandas as pd
        self.seg_df = pd.read_csv(segments_csv)
        self.seg_df = self.seg_df[self.seg_df["modality"] == "EEG"]

    def __len__(self) -> int:
        return len(self.seg_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.seg_df.iloc[idx]
        start = int(row["start"])
        end = int(row["end"])
        segment = self.data[:, start : end + 1]  # [C, T]
        x = torch.tensor(segment, dtype=torch.float32)
        y = torch.tensor(row["bought"] if "bought" in row else -1, dtype=torch.float32)
        return x, y


def load_eeg_segments(
    npy_path: Path,
    segments_csv: Path,
    batch_size: int,
    channels: List[int] = None,
    shuffle: bool = True,
) -> DataLoader:
    ds = EEGSegmentsDataset(npy_path, segments_csv, channels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

