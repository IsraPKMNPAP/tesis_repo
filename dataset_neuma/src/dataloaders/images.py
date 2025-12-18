from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class ProductImageDataset(Dataset):
    def __init__(
        self,
        products_csv: Path,
        transform: Optional[Callable] = None,
        label_col: str = "bought",
    ) -> None:
        self.df = pd.read_csv(products_csv)
        self.df = self.df.dropna(subset=["image_path", label_col])
        self.transform = transform
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(Path(row["image_path"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = torch.tensor(row[self.label_col], dtype=torch.float32)
        return img, y


def load_product_images(
    products_csv: Path,
    transform: Optional[Callable],
    batch_size: int,
    label_col: str = "bought",
    shuffle: bool = True,
) -> DataLoader:
    ds = ProductImageDataset(products_csv, transform=transform, label_col=label_col)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

