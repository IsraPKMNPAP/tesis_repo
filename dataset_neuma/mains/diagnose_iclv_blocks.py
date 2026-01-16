from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from utils.features import load_features_file


def load_cols(path: Path | None) -> List[str]:
    if path is None:
        return []
    if not path.exists():
        return []
    return [c.strip().lower() for c in load_features_file(path)]


def corr_report(df: pd.DataFrame, cols: List[str], label_col: str, name: str, threshold: float) -> None:
    if not cols:
        print(f"[{name}] no columns.")
        return
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr()
    # upper triangle pairs
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iat[i, j]
            if pd.notna(val) and abs(val) >= threshold:
                pairs.append((cols[i], cols[j], float(val)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"[{name}] high |corr| >= {threshold}: {len(pairs)} pairs")
    for a, b, v in pairs[:20]:
        print(f"  {a} vs {b}: {v:.4f}")

    if label_col in df.columns:
        y = pd.to_numeric(df[label_col], errors="coerce")
        corr_y = sub.apply(lambda s: s.corr(y))
        corr_y = corr_y.dropna().sort_values(key=lambda s: s.abs(), ascending=False)
        print(f"[{name}] corr with label (top 15):")
        for c, v in corr_y.head(15).items():
            print(f"  {c}: {float(v):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlaciones por bloque ICLV (clasico y multimodal).")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--label-col", type=str, default="bought")
    parser.add_argument("--obs-lt", type=Path, required=True)
    parser.add_argument("--obs-u", type=Path, required=True)
    parser.add_argument("--obs-i", type=Path, required=False)
    parser.add_argument("--mm-obs-lt", type=Path, required=False)
    parser.add_argument("--mm-obs-u", type=Path, required=False)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.lower()
    label_col = args.label_col.lower()

    obs_lt = load_cols(args.obs_lt)
    obs_u = load_cols(args.obs_u)
    obs_i = load_cols(args.obs_i) if args.obs_i else []
    mm_lt = load_cols(args.mm_obs_lt) if args.mm_obs_lt else []
    mm_u = load_cols(args.mm_obs_u) if args.mm_obs_u else []

    corr_report(df, obs_lt, label_col, "iclv_obs_lt", args.threshold)
    corr_report(df, obs_u, label_col, "iclv_obs_u", args.threshold)
    if obs_i:
        corr_report(df, obs_i, label_col, "iclv_obs_i", args.threshold)
    if mm_lt:
        corr_report(df, mm_lt, label_col, "mm_obs_lt", args.threshold)
    if mm_u:
        corr_report(df, mm_u, label_col, "mm_obs_u", args.threshold)


if __name__ == "__main__":
    main()
