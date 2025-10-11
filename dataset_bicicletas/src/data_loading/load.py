from __future__ import annotations

import os
from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    """Return the repository root (directory containing dataset_bicicletas)."""
    # Assuming this file lives at <root>/dataset_bicicletas/src/data_loading/load.py
    return Path(__file__).resolve().parents[3]


def default_data_path() -> Path:
    """Default CSV path: dataset_bicicletas/data/raw/all_data.csv relative to repo root."""
    return get_project_root() / "dataset_bicicletas" / "data" / "raw" / "all_data.csv"


def load_all_data(csv_path: os.PathLike | str | None = None) -> pd.DataFrame:
    """Load the bicicletas dataset from CSV.

    Args:
        csv_path: Optional explicit path to the CSV. If None, uses default_data_path().

    Returns:
        DataFrame with the raw data loaded.
    """
    path = Path(csv_path) if csv_path else default_data_path()
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}")
    return pd.read_csv(path)

