from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def timestamp_suffix() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def default_prefix(csv_path: str | Path, label: str) -> str:
    stem = Path(csv_path).stem
    return f"{stem}_{label}_{timestamp_suffix()}"


def save_text(text: str, out_path: str | Path):
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    out_path.write_text(text, encoding="utf-8")


def save_probs(
    probs: np.ndarray,
    classes: Sequence[str] | Sequence[int],
    out_path: str | Path,
    index: Optional[pd.Index] = None,
):
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    df = pd.DataFrame(probs, columns=list(classes))
    if index is not None:
        df.index = index
    df.to_csv(out_path, index=True)


def save_model_pickle(model, out_path: str | Path):
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    joblib.dump(model, out_path)

