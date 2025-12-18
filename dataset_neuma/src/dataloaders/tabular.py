from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class TabularDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cat_cols: List[str],
        num_cols: List[str],
        label_col: str,
        ohe: Optional[OneHotEncoder] = None,
        scaler: Optional[StandardScaler] = None,
    ) -> None:
        self.label = torch.tensor(df[label_col].to_numpy(), dtype=torch.float32)

        if ohe is None:
            try:
                self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            except TypeError:
                # scikit-learn >=1.2 usa sparse_output en lugar de sparse
                self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            self.ohe = ohe
        cat_data = self.ohe.fit_transform(df[cat_cols]) if ohe is None else self.ohe.transform(df[cat_cols])

        self.scaler = scaler or StandardScaler()
        num_data = self.scaler.fit_transform(df[num_cols]) if scaler is None else self.scaler.transform(df[num_cols])

        self.x = torch.tensor(np.hstack([cat_data, num_data]), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.label[idx]


def load_tabular(
    csv_path: Path,
    cat_cols: List[str],
    num_cols: List[str],
    label_col: str,
    batch_size: int,
    shuffle: bool = True,
) -> Tuple[DataLoader, OneHotEncoder, StandardScaler, int]:
    df = pd.read_csv(csv_path)
    # Normaliza nombres de columnas a minúsculas
    df.columns = df.columns.str.lower()
    cat_cols_l = [c.lower() for c in cat_cols]
    num_cols_l = [c.lower() for c in num_cols]
    label_col_l = label_col.lower()

    ds = TabularDataset(df, cat_cols_l, num_cols_l, label_col_l)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    input_dim = ds.x.shape[1]
    return loader, ds.ohe, ds.scaler, input_dim


def save_preprocessors(ohe: OneHotEncoder, scaler: StandardScaler, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ohe.json", "w", encoding="utf-8") as f:
        json.dump({"categories": [list(cat) for cat in ohe.categories_]}, f, ensure_ascii=False, indent=2)
    np.save(out_dir / "scaler_mean.npy", scaler.mean_)
    np.save(out_dir / "scaler_scale.npy", scaler.scale_)
