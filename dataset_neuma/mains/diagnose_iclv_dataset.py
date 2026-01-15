from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from utils.features import load_features_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostico de escala, colinealidad y leakage para ICLV.")
    parser.add_argument("--data", type=Path, default=Path("./data/processed/multimodal_join.csv"))
    parser.add_argument("--obs-lt-cols", type=Path, default=Path("./utils/columns/iclv/obs_lt.txt"))
    parser.add_argument("--obs-u-cols", type=Path, default=Path("./utils/columns/iclv/obs_u.txt"))
    parser.add_argument("--label-col", type=str, default="bought")
    parser.add_argument("--cat-unique-threshold", type=int, default=50)
    parser.add_argument("--corr-thresh", type=float, default=0.6)
    parser.add_argument("--leak-thresh", type=float, default=0.7)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.lower()
    label_col = args.label_col.lower()

    obs_lt = [c.strip().lower() for c in load_features_file(args.obs_lt_cols)]
    obs_u = [c.strip().lower() for c in load_features_file(args.obs_u_cols)]
    cols = [c for c in obs_lt + obs_u if c in df.columns]

    print(f"Total cols used: {len(cols)}")
    summary = []
    for c in cols:
        s = df[c]
        nunique = s.nunique(dropna=True)
        dtype = s.dtype
        missing = s.isna().sum()
        summary.append({"col": c, "dtype": str(dtype), "nunique": int(nunique), "missing": int(missing)})
    summary_df = pd.DataFrame(summary)
    print(summary_df.head(10).to_string(index=False))

    # Numeric-only correlation
    num_cols = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0 and s.std(skipna=True) > 0:
            num_cols.append(c)
    if num_cols:
        corr = df[num_cols].apply(pd.to_numeric, errors="coerce").corr()
        high_pairs = []
        for i, c1 in enumerate(num_cols):
            for j, c2 in enumerate(num_cols):
                if j <= i:
                    continue
                val = corr.iloc[i, j]
                if pd.notna(val) and abs(val) >= args.corr_thresh:
                    high_pairs.append((c1, c2, float(val)))
        if high_pairs:
            high_pairs = sorted(high_pairs, key=lambda x: abs(x[2]), reverse=True)
            print("\nHigh colinearity pairs (|corr|>=%.2f):" % args.corr_thresh)
            for c1, c2, v in high_pairs[:20]:
                print(f"  {c1} vs {c2}: {v:.3f}")

        if label_col in df.columns:
            y = pd.to_numeric(df[label_col], errors="coerce")
            leak = []
            for c in num_cols:
                corr_y = pd.to_numeric(df[c], errors="coerce").corr(y)
                if pd.notna(corr_y) and abs(corr_y) >= args.leak_thresh:
                    leak.append((c, float(corr_y)))
            if leak:
                leak = sorted(leak, key=lambda x: abs(x[1]), reverse=True)
                print("\nPotential leakage (|corr with label|>=%.2f):" % args.leak_thresh)
                for c, v in leak[:20]:
                    print(f"  {c}: {v:.3f}")


if __name__ == "__main__":
    main()
