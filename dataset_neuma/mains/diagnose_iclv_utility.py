from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from utils.features import load_features_file


def normalize_subject(val: str) -> str:
    s = str(val).strip().lower()
    digits = "".join([c for c in s if c.isdigit()])
    if digits:
        return str(int(digits))
    return s


def preprocess_block(train_df: pd.DataFrame, cols: List[str], prefix: str, cat_unique_threshold: int) -> pd.DataFrame:
    import pandas.api.types as ptypes

    num_cols = [c for c in cols if ptypes.is_numeric_dtype(train_df[c])]
    cat_cols = [c for c in cols if c not in num_cols]

    # auto-categorize low-cardinality numerics
    for c in list(num_cols):
        if train_df[c].nunique(dropna=True) <= cat_unique_threshold:
            num_cols.remove(c)
            cat_cols.append(c)

    out_parts = []
    if num_cols:
        means = train_df[num_cols].mean()
        stds = train_df[num_cols].std().replace(0, 1)
        tr_num = (train_df[num_cols] - means) / stds
        tr_num.columns = [f"{prefix}{c}" for c in num_cols]
        out_parts.append(tr_num)
    if cat_cols:
        tr_cat = pd.get_dummies(train_df[cat_cols].astype(str), prefix=[f"{prefix}{c}" for c in cat_cols])
        out_parts.append(tr_cat)
    return pd.concat(out_parts, axis=1) if out_parts else pd.DataFrame(index=train_df.index)


def vif_matrix(X: np.ndarray, max_cols: int = 200) -> List[tuple[str, float]]:
    # simple VIF using linear regression on numeric matrix
    from sklearn.linear_model import LinearRegression

    n, p = X.shape
    if p == 0:
        return []
    if p > max_cols:
        return [("vif_skipped_too_many_cols", float(p))]
    vifs = []
    for i in range(p):
        y = X[:, i]
        X_ = np.delete(X, i, axis=1)
        if X_.shape[1] == 0 or np.std(y) == 0:
            vifs.append(np.inf)
            continue
        reg = LinearRegression().fit(X_, y)
        r2 = reg.score(X_, y)
        vif = 1.0 / max(1e-6, (1.0 - r2))
        vifs.append(vif)
    return vifs


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostico de utilidad ICLV (condición, VIF, varianza).")
    parser.add_argument("--data", type=Path, default=Path("./data/processed/multimodal_join.csv"))
    parser.add_argument("--obs-u-cols", type=Path, default=Path("./utils/columns/iclv/obs_u.txt"))
    parser.add_argument("--label-col", type=str, default="bought")
    parser.add_argument("--cat-unique-threshold", type=int, default=50)
    parser.add_argument("--max-vif-cols", type=int, default=200)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.lower()
    if "subject" in df.columns:
        df["subject"] = df["subject"].apply(normalize_subject)

    obs_u_cols = [c.strip().lower() for c in load_features_file(args.obs_u_cols)]
    obs_u_cols = [c for c in obs_u_cols if c in df.columns]
    if not obs_u_cols:
        print("No obs_u cols found in data.")
        return

    X = preprocess_block(df, obs_u_cols, prefix="u_", cat_unique_threshold=args.cat_unique_threshold)
    print("obs_u design matrix shape:", X.shape)
    # variance distribution
    var = X.var(axis=0)
    print("Var stats: min=%.6f median=%.6f max=%.6f" % (var.min(), var.median(), var.max()))
    low_var = var[var < 1e-6].index.tolist()
    if low_var:
        print("Near-zero variance cols:", low_var[:20])

    # condition number
    if X.shape[1] > 0:
        cov = np.cov(X.to_numpy(dtype=float), rowvar=False)
        if cov.size > 0:
            eigvals = np.linalg.eigvalsh(cov)
            cond = np.inf if np.min(eigvals) == 0 else float(np.max(eigvals) / np.min(eigvals))
            print("Condition number (cov):", cond)

    # VIF
    if X.shape[1] > 0:
        vifs = vif_matrix(X.to_numpy(dtype=float), max_cols=args.max_vif_cols)
        if vifs and isinstance(vifs[0], str):
            print(vifs[0])
        else:
            vif_series = pd.Series(vifs, index=X.columns).sort_values(ascending=False)
            print("Top 20 VIF:")
            print(vif_series.head(20).to_string())

    # corr with label
    if args.label_col.lower() in df.columns:
        y = pd.to_numeric(df[args.label_col.lower()], errors="coerce")
        corr = X.apply(lambda s: pd.to_numeric(s, errors="coerce").corr(y))
        high = corr[abs(corr) >= 0.7].sort_values(key=lambda s: s.abs(), ascending=False)
        if not high.empty:
            print("High corr with label (>=0.7):")
            print(high.head(20).to_string())


if __name__ == "__main__":
    main()
