from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


def split_by_participant(
    df: pd.DataFrame,
    participant_col: str = "participant",
    val_frac: float = 0.2,
    test_frac: float = 0.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Split train/val/test stratified by participant IDs (no leakage).

    Fractions are over distinct participants. Returns (train, val, test, info_dict).
    """
    if participant_col not in df.columns:
        raise KeyError(f"Falta la columna '{participant_col}' en el dataframe.")
    parts = df[participant_col].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(parts)
    n_parts = len(parts)
    n_val = int(round(n_parts * val_frac))
    n_test = int(round(n_parts * test_frac))
    val_parts = set(parts[:n_val])
    test_parts = set(parts[n_val : n_val + n_test])
    train_parts = set(parts[n_val + n_test :])

    def _filter(parts_set):
        return df[df[participant_col].isin(parts_set)].reset_index(drop=True)

    df_train = _filter(train_parts)
    df_val = _filter(val_parts)
    df_test = _filter(test_parts)

    info = {
        "n_parts_total": n_parts,
        "train_parts": sorted(train_parts),
        "val_parts": sorted(val_parts),
        "test_parts": sorted(test_parts),
        "n_train_parts": len(train_parts),
        "n_val_parts": len(val_parts),
        "n_test_parts": len(test_parts),
        "n_train_obs": len(df_train),
        "n_val_obs": len(df_val),
        "n_test_obs": len(df_test),
    }
    return df_train, df_val, df_test, info


def format_split_report(info: dict) -> str:
    lines = [
        f"Participantes total: {info['n_parts_total']}",
        f"Train: {info['n_train_parts']} parts, {info['n_train_obs']} obs -> {info['train_parts']}",
        f"Val:   {info['n_val_parts']} parts, {info['n_val_obs']} obs -> {info['val_parts']}",
        f"Test:  {info['n_test_parts']} parts, {info['n_test_obs']} obs -> {info['test_parts']}",
    ]
    return "\n".join(lines)
