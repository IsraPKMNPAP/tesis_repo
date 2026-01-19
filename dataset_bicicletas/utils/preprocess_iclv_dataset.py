from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


ACTION_MAP = {
    "accelerate": 0,
    "brake": 1,
    "decelerate": 2,
    "maintain speed": 3,
    "wait": 4,
}


def load_cols(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [c.strip() for c in path.read_text(encoding="utf-8").splitlines() if c.strip()]


def drop_rare_or_weird(df: pd.DataFrame, cols: Sequence[str], null_thresh: int) -> Tuple[List[str], List[str]]:
    dropped: List[str] = []
    kept: List[str] = []
    for c in cols:
        if c not in df.columns:
            dropped.append(c)
            continue
        series = df[c]
        if series.map(lambda v: isinstance(v, (list, dict, tuple, set))).any():
            dropped.append(c)
            continue
        if series.isna().sum() > null_thresh:
            dropped.append(c)
            continue
        kept.append(c)
    return kept, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocesa dataset para ICLV (validación y filtrado).")
    ap.add_argument("--input", type=Path, default=Path("data/raw/all_data.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/iclv_df.csv"))
    ap.add_argument("--obs-lt", type=Path, default=Path("utils/feature_sets/obs_lt.txt"))
    ap.add_argument("--obs-u", type=Path, default=Path("utils/feature_sets/obs_u.txt"))
    ap.add_argument("--obs-i", type=Path, default=Path("utils/feature_sets/obs_i.txt"))
    ap.add_argument("--label-col", type=str, default="action_proc")
    ap.add_argument("--action-col", type=str, default="action")
    ap.add_argument("--participant-col", type=str, default="participant")
    ap.add_argument("--timestamp-col", type=str, default="timestamp")
    ap.add_argument("--null-thresh", type=int, default=1000, help="Max nulos permitidos por columna.")
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    obs_lt_cols = load_cols(args.obs_lt)
    obs_u_cols = load_cols(args.obs_u)
    obs_i_cols = load_cols(args.obs_i)

    missing = {
        "obs_lt": [c for c in obs_lt_cols if c not in df.columns],
        "obs_u": [c for c in obs_u_cols if c not in df.columns],
        "obs_i": [c for c in obs_i_cols if c not in df.columns],
    }
    for k, v in missing.items():
        if v:
            print(f"[warn] {k} missing cols: {v}")

    # Build label
    if args.label_col not in df.columns and args.action_col in df.columns:
        df[args.label_col] = df[args.action_col].map(ACTION_MAP)
    if df[args.label_col].dtype == object:
        df[args.label_col] = df[args.label_col].map(ACTION_MAP)
    df = df.dropna(subset=[args.label_col]).reset_index(drop=True)
    df[args.label_col] = df[args.label_col].astype(int)

    # Drop columns not in txt lists before any processing
    keep_cols = sorted(set(obs_lt_cols + obs_u_cols + obs_i_cols))
    meta_cols = [c for c in [args.label_col, args.participant_col, args.timestamp_col] if c in df.columns]
    df = df[meta_cols + keep_cols].copy()
    if args.action_col in df.columns:
        df = df.drop(columns=[args.action_col])

    # Drop columns with too many nulls or weird types
    lt_keep, lt_drop = drop_rare_or_weird(df, obs_lt_cols, args.null_thresh)
    u_keep, u_drop = drop_rare_or_weird(df, obs_u_cols, args.null_thresh)
    i_keep, i_drop = drop_rare_or_weird(df, obs_i_cols, args.null_thresh)
    if lt_drop:
        print(f"[drop] obs_lt (null/type): {lt_drop}")
    if u_drop:
        print(f"[drop] obs_u (null/type): {u_drop}")
    if i_drop:
        print(f"[drop] obs_i (null/type): {i_drop}")

    out = df[meta_cols + lt_keep + u_keep + i_keep].copy()
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[OK] saved: {out_path} rows={len(out)} cols={out.shape[1]}")


if __name__ == "__main__":
    main()
