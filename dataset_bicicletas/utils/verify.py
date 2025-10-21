from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
from typing import List, Optional

import pandas as pd
import torch


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p))


def print_df_overview(df: pd.DataFrame, show_dtypes: bool = False):
    print("=== DataFrame Overview ===")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Columns:")
    print(", ".join(list(df.columns)))
    if show_dtypes:
        print("\nDtypes:")
        print(df.dtypes)


def print_sample_rows(df: pd.DataFrame, columns: List[str], head: int = 5):
    cols_present = [c for c in columns if c in df.columns]
    if not cols_present:
        print("No requested columns found to display.")
        return
    print(f"\n=== Head ({head}) of selected columns ===")
    print(df[cols_present].head(head).to_string(index=False))


def label_stats(df: pd.DataFrame, label_col: Optional[str]):
    if not label_col or label_col not in df.columns:
        print("\n(No label_col provided or not found in DataFrame)")
        return
    s = df[label_col]
    print(f"\n=== Label stats for '{label_col}' ===")
    print("Non-null count:", int(s.notna().sum()))
    vc = s.value_counts(dropna=True)
    # Limit to top 20 if many classes
    if len(vc) > 20:
        print(vc.head(20))
        print(f"(and {len(vc)-20} more)")
    else:
        print(vc)


def check_paths_and_probe_tensors(
    df: pd.DataFrame,
    path_col: str,
    sample_pt: int = 3,
    random_sample: bool = False,
):
    if path_col not in df.columns:
        print(f"\n[WARN] Column '{path_col}' not found in DataFrame. Skipping tensor probing.")
        return
    paths = df[path_col].dropna().astype(str).tolist()
    if not paths:
        print(f"\n[WARN] No paths in '{path_col}'.")
        return
    exists = [Path(p).exists() for p in paths]
    missing = len(paths) - sum(exists)
    print(f"\n=== Path existence check for '{path_col}' ===")
    print(f"Total non-null paths: {len(paths)} | Missing files: {missing}")
    if missing > 0:
        # Show up to 5 missing examples
        miss_examples = [p for p, ok in zip(paths, exists) if not ok][:5]
        if miss_examples:
            print("Missing examples:", miss_examples)

    # Probe a few .pt files
    k = min(sample_pt, len(paths))
    if k <= 0:
        return
    cand = [p for p, ok in zip(paths, exists) if ok]
    if not cand:
        print("No existing files to probe.")
        return
    if random_sample:
        sample = random.sample(cand, k)
    else:
        sample = cand[:k]

    print(f"\n=== Probing {len(sample)} tensor files ===")
    for p in sample:
        pp = Path(p)
        print(f"\nFile: {pp}")
        try:
            obj = torch.load(pp, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[ERROR] torch.load failed: {e}")
            continue
        if isinstance(obj, dict):
            keys = list(obj.keys())
            print("Type: dict | Keys:", keys)
            frames = obj.get("frames") or obj.get("x")
            if isinstance(frames, torch.Tensor):
                print("frames shape:", tuple(frames.shape), "dtype:", frames.dtype)
            lbl = obj.get("label") or obj.get("y")
            if isinstance(lbl, torch.Tensor):
                try:
                    lbl = int(lbl.item())
                except Exception:
                    pass
            print("label:", lbl)
            print("timestamp:", obj.get("timestamp") or obj.get("ts"))
            print("window_id:", obj.get("window_id") or obj.get("idx"))
            print("participant:", obj.get("participant"))
        elif isinstance(obj, torch.Tensor):
            print("Type: Tensor | shape:", tuple(obj.shape), "dtype:", obj.dtype)
        else:
            print("Type:", type(obj))


def main():
    ap = argparse.ArgumentParser(description="Verify DataFrame columns and probe tensor .pt contents for the video pipeline")
    ap.add_argument("--pickle", type=str, required=True, help="Ruta al pickle a verificar (raw o processed)")
    ap.add_argument("--path-col", type=str, default="gpu_tensor_path", help="Columna con rutas a .pt")
    ap.add_argument("--label-col", type=str, default="action", help="Columna de labels en el DataFrame (ej. action_proc)")
    ap.add_argument("--timestamp-col", type=str, default="timestamp")
    ap.add_argument("--window-id-col", type=str, default="window")
    ap.add_argument("--head", type=int, default=5, help="Filas a mostrar de ejemplo")
    ap.add_argument("--show-dtypes", action="store_true", help="Mostrar dtypes del DataFrame")
    ap.add_argument("--sample-pt", type=int, default=3, help="Cantidad de archivos .pt a probar")
    ap.add_argument("--random", action="store_true", help="Tomar muestras aleatorias de .pt para probeo")

    args = ap.parse_args()

    pkl = _expand(args.pickle)
    if not pkl.exists():
        raise FileNotFoundError(f"No existe el pickle: {pkl}")
    df = pd.read_pickle(pkl)

    print_df_overview(df, show_dtypes=args.show_dtypes)

    cols_to_show = [c for c in [
        "participant",
        args.timestamp_col,
        args.window_id_col,
        args.path_col,
        args.label_col,
        "session_id",
        "is_imputed",
    ] if c in df.columns]
    print_sample_rows(df, cols_to_show, head=args.head)

    label_stats(df, args.label_col)

    check_paths_and_probe_tensors(df, args.path_col, sample_pt=args.sample_pt, random_sample=args.random)


if __name__ == "__main__":
    main()

