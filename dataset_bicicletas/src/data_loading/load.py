from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_all_data(csv_path: str | Path) -> pd.DataFrame:
    """Load dataset from an explicit CSV path.

    Notes:
        - `csv_path` is required. Use relative paths from the working directory
          (e.g., run from `dataset_bicicletas` and pass `data/processed/dataset_bicicletas_clean.csv`).
    """
    if not csv_path:
        raise ValueError("csv_path es obligatorio. Proporcione la ruta al CSV.")
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el CSV en: {path}")
    return pd.read_csv(path)
