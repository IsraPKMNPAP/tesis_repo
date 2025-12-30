from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGAutoencoderDataset(Dataset):
    """Dataset para autoencoder EEG usando archivos npy (concat)."""

    def __init__(self, paths, eeg_len: int):
        self.paths = list(paths)
        self.eeg_len = eeg_len

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        arr = np.load(path)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        x = torch.tensor(arr, dtype=torch.float32)
        # recortar/pad a eeg_len
        if x.shape[1] > self.eeg_len:
            x = x[:, : self.eeg_len]
        elif x.shape[1] < self.eeg_len:
            pad = self.eeg_len - x.shape[1]
            x = torch.cat([x, torch.zeros((x.shape[0], pad), dtype=x.dtype)], dim=1)
        return x, path
