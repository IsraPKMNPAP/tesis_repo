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


def coerce_block(
    df: pd.DataFrame,
    cols: Sequence[str],
    cat_unique_threshold: int,
    block_name: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Return (processed_block_df, numeric_cols, dropped_cols)."""
    dropped: List[str] = []
    numeric_cols: List[str] = []
    cat_cols: List[str] = []

    for c in cols:
        if c not in df.columns:
            dropped.append(c)
            continue
        series = df[c]
        if series.map(lambda v: isinstance(v, (list, dict, tuple, set))).any():
            dropped.append(c)
            continue
        try:
            nunique = series.nunique(dropna=True)
        except Exception:
            dropped.append(c)
            continue
        if nunique < cat_unique_threshold:
            cat_cols.append(c)
        else:
            numeric_cols.append(c)

    out = pd.DataFrame(index=df.index)
    # Numeric: force float64
    for c in numeric_cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() == 0:
            dropped.append(c)
            continue
        out[c] = vals.astype(np.float64)

    # Categorical: fill missing and one-hot later
    for c in cat_cols:
        out[c] = df[c].astype(str).fillna("missing")

    if dropped:
        print(f"[drop] {block_name} removed columns: {dropped}")
    return out, numeric_cols, dropped


def standardize_numeric(df: pd.DataFrame, numeric_cols: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    stats: Dict[str, Tuple[float, float]] = {}
    out = df.copy()
    for c in numeric_cols:
        col = out[c].astype(np.float64)
        mu = float(col.mean())
        sd = float(col.std(ddof=0))
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        out[c] = (col - mu) / sd
        stats[c] = (mu, sd)
    return out, stats


def one_hot(df: pd.DataFrame, cat_cols: Sequence[str], prefix: str) -> pd.DataFrame:
    if not cat_cols:
        return pd.DataFrame(index=df.index)
    return pd.get_dummies(df[cat_cols].astype(str), prefix=[f"{prefix}{c}" for c in cat_cols], drop_first=True)


def drop_high_corr(X: pd.DataFrame, thresh: float) -> Tuple[pd.DataFrame, List[str]]:
    if X.shape[1] <= 1:
        return X, []
    corr = X.corr().abs().fillna(0.0)
    keep = list(X.columns)
    dropped: List[str] = []
    while True:
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        max_val = upper.max().max()
        if not np.isfinite(max_val) or max_val <= thresh:
            break
        # drop the column with the largest mean correlation
        mean_corr = corr.mean()
        drop_col = mean_corr.idxmax()
        keep.remove(drop_col)
        dropped.append(drop_col)
        corr = corr.loc[keep, keep]
    return X[keep], dropped


def drop_by_label_corr(X: pd.DataFrame, y: pd.Series, thresh: float) -> Tuple[pd.DataFrame, List[str]]:
    if X.shape[1] == 0:
        return X, []
    y_num = pd.to_numeric(y, errors="coerce")
    corrs = X.apply(lambda c: np.corrcoef(c, y_num)[0, 1] if c.nunique(dropna=True) > 1 else 0.0)
    corrs = corrs.abs().fillna(0.0)
    drop_cols = corrs[corrs > thresh].index.tolist()
    if drop_cols:
        return X.drop(columns=drop_cols), drop_cols
    return X, []


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
    ap.add_argument("--cat-unique-threshold", type=int, default=10)
    ap.add_argument("--corr-block-thresh", type=float, default=0.6)
    ap.add_argument("--corr-label-thresh", type=float, default=0.7)
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

    keep_meta = [c for c in [args.action_col, args.label_col, args.participant_col, args.timestamp_col] if c in df.columns]

    # Process each block
    lt_df, lt_num, lt_drop = coerce_block(df, obs_lt_cols, args.cat_unique_threshold, "obs_lt")
    u_df, u_num, u_drop = coerce_block(df, obs_u_cols, args.cat_unique_threshold, "obs_u")
    i_df, i_num, i_drop = coerce_block(df, obs_i_cols, args.cat_unique_threshold, "obs_i")

    lt_df, lt_stats = standardize_numeric(lt_df, lt_num)
    u_df, u_stats = standardize_numeric(u_df, u_num)
    i_df, i_stats = standardize_numeric(i_df, i_num)
    print(f"[stats] numeric standardized: lt={len(lt_num)} u={len(u_num)} i={len(i_num)}")

    lt_cat = [c for c in lt_df.columns if c not in lt_num]
    u_cat = [c for c in u_df.columns if c not in u_num]
    i_cat = [c for c in i_df.columns if c not in i_num]

    lt_onehot = one_hot(lt_df, lt_cat, "lt_")
    u_onehot = one_hot(u_df, u_cat, "u_")
    i_onehot = one_hot(i_df, i_cat, "i_")

    lt_block = pd.concat([lt_df[lt_num], lt_onehot], axis=1)
    u_block = pd.concat([u_df[u_num], u_onehot], axis=1)
    i_block = pd.concat([i_df[i_num], i_onehot], axis=1)

    # Correlation filters within each block
    lt_block, lt_corr_drop = drop_high_corr(lt_block, args.corr_block_thresh)
    u_block, u_corr_drop = drop_high_corr(u_block, args.corr_block_thresh)
    i_block, i_corr_drop = drop_high_corr(i_block, args.corr_block_thresh)
    if lt_corr_drop:
        print(f"[drop] obs_lt high-corr cols: {lt_corr_drop}")
    if u_corr_drop:
        print(f"[drop] obs_u high-corr cols: {u_corr_drop}")
    if i_corr_drop:
        print(f"[drop] obs_i high-corr cols: {i_corr_drop}")

    # Label correlation (all features)
    all_block = pd.concat([lt_block, u_block, i_block], axis=1)
    all_block, label_drop = drop_by_label_corr(all_block, df[args.label_col], args.corr_label_thresh)
    if label_drop:
        print(f"[drop] label-correlated cols (> {args.corr_label_thresh}): {label_drop}")

    # Re-split after label drop
    lt_cols_final = [c for c in lt_block.columns if c in all_block.columns]
    u_cols_final = [c for c in u_block.columns if c in all_block.columns]
    i_cols_final = [c for c in i_block.columns if c in all_block.columns]

    out = pd.concat(
        [
            df[keep_meta].reset_index(drop=True),
            all_block[lt_cols_final + u_cols_final + i_cols_final].reset_index(drop=True),
        ],
        axis=1,
    )
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[OK] saved: {out_path} rows={len(out)} cols={out.shape[1]}")


if __name__ == "__main__":
    main()
