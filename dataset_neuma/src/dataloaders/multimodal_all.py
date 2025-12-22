from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader


def _crop_or_pad(arr: np.ndarray, target_len: int) -> np.ndarray:
    c, t = arr.shape
    if t == target_len:
        return arr
    if t < target_len:
        out = np.zeros((c, target_len), dtype=arr.dtype)
        out[:, :t] = arr
        return out
    start = max(0, (t - target_len) // 2)
    end = start + target_len
    return arr[:, start:end]


class MultimodalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cat_cols: List[str],
        num_cols: List[str],
        label_col: str,
        eeg_len: int,
        cache_clip: bool = True,
        cache_eeg: bool = True,
        ohe: Optional[OneHotEncoder] = None,
        scaler: Optional[StandardScaler] = None,
    ) -> None:
        df = df.copy()
        df.columns = df.columns.str.lower()
        cat_cols = [c.lower() for c in cat_cols]
        num_cols = [c.lower() for c in num_cols]
        label_col = label_col.lower()

        # Tabular preprocess
        if ohe is None:
            try:
                self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            except TypeError:
                self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            cat_data = self.ohe.fit_transform(df[cat_cols])
        else:
            self.ohe = ohe
            cat_data = self.ohe.transform(df[cat_cols])

        if scaler is None:
            self.scaler = StandardScaler()
            num_data = self.scaler.fit_transform(df[num_cols])
        else:
            self.scaler = scaler
            num_data = self.scaler.transform(df[num_cols])

        self.tab = torch.tensor(np.hstack([cat_data, num_data]), dtype=torch.float32)
        self.labels = torch.tensor(df[label_col].to_numpy(), dtype=torch.float32)

        self.clip_paths = df["embedding_path"].tolist()
        self.eeg_paths = df["eeg_concat_path"].tolist()
        self.eeg_len = eeg_len
        self.cache_clip = cache_clip
        self.cache_eeg = cache_eeg
        self.clip_cache: Dict[str, np.ndarray] = {} if cache_clip else None
        self.eeg_cache: Dict[str, np.ndarray] = {} if cache_eeg else None

    def __len__(self) -> int:
        return len(self.labels)

    def _load_clip(self, path: str) -> np.ndarray:
        if self.cache_clip:
            if path not in self.clip_cache:
                self.clip_cache[path] = np.load(path)
            return self.clip_cache[path]
        return np.load(path)

    def _load_eeg(self, path: str) -> np.ndarray:
        if self.cache_eeg:
            if path not in self.eeg_cache:
                self.eeg_cache[path] = np.load(path)
            return self.eeg_cache[path]
        return np.load(path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tab = self.tab[idx]
        clip = torch.tensor(self._load_clip(self.clip_paths[idx]), dtype=torch.float32)
        eeg_arr = self._load_eeg(self.eeg_paths[idx])
        eeg_arr = _crop_or_pad(eeg_arr, self.eeg_len)
        eeg = torch.tensor(eeg_arr, dtype=torch.float32)
        y = self.labels[idx]
        return tab, clip, eeg, y


def prepare_feature_lists(df: pd.DataFrame, label_col: str) -> Tuple[List[str], List[str]]:
    df = df.copy()
    df.columns = df.columns.str.lower()
    label_col = label_col.lower()
    exclude = {
        "subject",
        "subject_norm",
        "page",
        "page_num",
        "product_id",
        "prod_num",
        "embedding_path",
        "eeg_concat_path",
        "eeg_shape",
        label_col,
    }
    cat_cols = []
    num_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if str(df[c].dtype) == "category" or df[c].dtype == object:
            cat_cols.append(c)
        elif np.issubdtype(df[c].dtype, np.number):
            num_cols.append(c)
    return cat_cols, num_cols


def load_multimodal(
    csv_path: Path,
    label_col: str,
    batch_size: int,
    eeg_len: int,
    cache_clip: bool = True,
    cache_eeg: bool = True,
    shuffle: bool = True,
    cat_cols: Optional[List[str]] = None,
    num_cols: Optional[List[str]] = None,
    ohe: Optional[OneHotEncoder] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[DataLoader, OneHotEncoder, StandardScaler, int, int, int]:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower()
    label_col = label_col.lower()
    if cat_cols is None or num_cols is None:
        cat_cols, num_cols = prepare_feature_lists(df, label_col)
    ds = MultimodalDataset(
        df=df,
        cat_cols=cat_cols,
        num_cols=num_cols,
        label_col=label_col,
        eeg_len=eeg_len,
        cache_clip=cache_clip,
        cache_eeg=cache_eeg,
        ohe=ohe,
        scaler=scaler,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    tab_dim = ds.tab.shape[1]
    clip_dim = len(np.load(ds.clip_paths[0]))
    eeg_ch = np.load(ds.eeg_paths[0]).shape[0]
    return loader, ds.ohe, ds.scaler, tab_dim, clip_dim, eeg_ch
