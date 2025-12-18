from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader


class TabularImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cat_cols: List[str],
        num_cols: List[str],
        label_col: str,
        ohe: Optional[OneHotEncoder] = None,
        scaler: Optional[StandardScaler] = None,
        cache_embeddings: bool = True,
    ) -> None:
        # Normaliza nombres a minúsculas
        df = df.copy()
        df.columns = df.columns.str.lower()
        cat_cols = [c.lower() for c in cat_cols]
        num_cols = [c.lower() for c in num_cols]
        label_col = label_col.lower()

        # Preprocesamiento tabular
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

        self.tabular = torch.tensor(np.hstack([cat_data, num_data]), dtype=torch.float32)
        self.labels = torch.tensor(df[label_col].to_numpy(), dtype=torch.float32)
        self.embedding_paths = df["embedding_path"].tolist()

        self.cache_embeddings = cache_embeddings
        self._cache: Dict[str, np.ndarray] = {} if cache_embeddings else None

    def __len__(self) -> int:
        return len(self.labels)

    def _load_embedding(self, path: str) -> np.ndarray:
        if self.cache_embeddings:
            if path not in self._cache:
                self._cache[path] = np.load(path)
            return self._cache[path]
        return np.load(path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tab = self.tabular[idx]
        emb = torch.tensor(self._load_embedding(self.embedding_paths[idx]), dtype=torch.float32)
        y = self.labels[idx]
        return tab, emb, y


def load_tabular_image(
    df: pd.DataFrame,
    cat_cols: List[str],
    num_cols: List[str],
    label_col: str,
    batch_size: int,
    cache_embeddings: bool = True,
    shuffle: bool = True,
    ohe: Optional[OneHotEncoder] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[DataLoader, OneHotEncoder, StandardScaler, int, int]:
    ds = TabularImageDataset(
        df=df,
        cat_cols=cat_cols,
        num_cols=num_cols,
        label_col=label_col,
        ohe=ohe,
        scaler=scaler,
        cache_embeddings=cache_embeddings,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    tab_dim = ds.tabular.shape[1]
    # Infer embedding dim from first sample
    emb_dim = len(np.load(ds.embedding_paths[0]))
    return loader, ds.ohe, ds.scaler, tab_dim, emb_dim


def save_preprocessors(ohe: OneHotEncoder, scaler: StandardScaler, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cats_serializable = []
    for cat in ohe.categories_:
        cats_serializable.append([c.item() if hasattr(c, "item") else c for c in cat])
    with open(out_dir / "ohe.json", "w", encoding="utf-8") as f:
        json.dump({"categories": cats_serializable}, f, ensure_ascii=False, indent=2)
    np.save(out_dir / "scaler_mean.npy", scaler.mean_)
    np.save(out_dir / "scaler_scale.npy", scaler.scale_)
