from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd


def split_by_subject(
    df: pd.DataFrame,
    subject_col: str = "subject",
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Split por sujeto (evita data leakage) con muestreo equiprobable de sujetos.
    Devuelve train_df, val_df y un dict con diagnóstico (listas de sujetos, tamaños).
    """
    if subject_col not in df.columns:
        raise ValueError(f"Columna {subject_col} no encontrada en DataFrame.")
    subjects = df[subject_col].dropna().unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_frac)))
    val_subj = set(subjects[:n_val])
    train_df = df[~df[subject_col].isin(val_subj)].reset_index(drop=True)
    val_df = df[df[subject_col].isin(val_subj)].reset_index(drop=True)
    info = {
        "subject_col": subject_col,
        "seed": seed,
        "val_frac": val_frac,
        "n_subjects": int(len(subjects)),
        "n_val_subjects": int(len(val_subj)),
        "val_subjects": sorted(map(str, val_subj)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
    }
    return train_df, val_df, info


def save_split_info(info: Dict, results_dir: Path, filename: str = "split_info.json") -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def split_by_subject_train_val_test(
    df: pd.DataFrame,
    subject_col: str = "subject",
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Split por sujeto en train/val/test."""
    if subject_col not in df.columns:
        raise ValueError(f"Columna {subject_col} no encontrada en DataFrame.")
    subjects = df[subject_col].dropna().unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_frac)))
    n_test = max(1, int(round(len(subjects) * test_frac)))
    val_subj = set(subjects[:n_val])
    test_subj = set(subjects[n_val : n_val + n_test])
    train_subj = set(subjects[n_val + n_test :])

    train_df = df[df[subject_col].isin(train_subj)].reset_index(drop=True)
    val_df = df[df[subject_col].isin(val_subj)].reset_index(drop=True)
    test_df = df[df[subject_col].isin(test_subj)].reset_index(drop=True)
    info = {
        "subject_col": subject_col,
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "n_subjects": int(len(subjects)),
        "n_train_subjects": int(len(train_subj)),
        "n_val_subjects": int(len(val_subj)),
        "n_test_subjects": int(len(test_subj)),
        "train_subjects": sorted(map(str, train_subj)),
        "val_subjects": sorted(map(str, val_subj)),
        "test_subjects": sorted(map(str, test_subj)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
    }
    return train_df, val_df, test_df, info
