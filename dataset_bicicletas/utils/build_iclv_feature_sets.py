from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _load_cols(path: str | None) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    return [c.strip() for c in p.read_text(encoding="utf-8").splitlines() if c.strip()]


def _read_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_pickle(path)


def _categorical_cols(df: pd.DataFrame, cols: List[str], max_unique: int) -> List[str]:
    out = []
    for c in cols:
        if c not in df.columns:
            continue
        try:
            nunique = df[c].nunique(dropna=True)
        except Exception:
            continue
        if nunique <= max_unique:
            out.append(c)
    return out


def _build_design(df: pd.DataFrame, cols: List[str], cat_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    base = df[cols].copy()
    base = base.replace([np.inf, -np.inf], np.nan)
    base = base.fillna(base.median(numeric_only=True))
    base_cat = base[cat_cols].astype(str) if cat_cols else pd.DataFrame(index=base.index)
    base_num = base.drop(columns=cat_cols, errors="ignore")

    dummies = pd.get_dummies(base_cat, drop_first=True) if not base_cat.empty else pd.DataFrame(index=base.index)
    X = pd.concat([base_num, dummies], axis=1)

    # map original -> expanded columns
    mapping: Dict[str, List[str]] = {c: [c] for c in base_num.columns}
    for c in cat_cols:
        mapping[c] = [d for d in dummies.columns if d.startswith(f"{c}_")]
    return X, mapping


def _drop_by_label_corr(
    X: pd.DataFrame,
    y: pd.Series,
    mapping: Dict[str, List[str]],
    thresh: float,
) -> List[str]:
    drop_cols = []
    y_num = pd.to_numeric(y, errors="coerce").fillna(0.0)
    for orig, cols in mapping.items():
        if not cols:
            continue
        vals = X[cols]
        corrs = []
        for c in cols:
            if vals[c].nunique(dropna=True) <= 1:
                corrs.append(0.0)
            else:
                corrs.append(abs(np.corrcoef(vals[c], y_num)[0, 1]))
        if corrs and np.nanmax(corrs) >= thresh:
            drop_cols.append(orig)
    return drop_cols


def _drop_by_feature_corr(X: pd.DataFrame, mapping: Dict[str, List[str]], thresh: float) -> List[str]:
    if X.shape[1] == 0:
        return []
    corr = X.corr().abs().fillna(0.0)
    keep = set(X.columns.tolist())
    dropped = set()

    for col in corr.columns:
        if col not in keep:
            continue
        high = corr.index[(corr[col] > thresh) & (corr.index != col)].tolist()
        for other in high:
            if other in keep:
                keep.remove(other)
                dropped.add(other)

    # map dropped expanded cols to original names
    dropped_orig = set()
    for orig, cols in mapping.items():
        if any(c in dropped for c in cols):
            dropped_orig.add(orig)
    return sorted(dropped_orig)


def _write_cols(cols: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(cols), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera nuevos obs_lt/obs_u/obs_i filtrados por correlacion.")
    ap.add_argument("--data", type=str, required=True, help="CSV o PKL con dataset.")
    ap.add_argument("--label-col", type=str, required=True)
    ap.add_argument("--obs-lt-cols-file", type=str, required=True)
    ap.add_argument("--obs-u-cols-file", type=str, required=True)
    ap.add_argument("--obs-i-cols-file", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--corr-thresh", type=float, default=0.7)
    ap.add_argument("--label-corr-thresh", type=float, default=0.7)
    ap.add_argument("--cat-max-unique", type=int, default=10)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    df = _read_df(Path(args.data)).reset_index(drop=True)
    if args.label_col not in df.columns:
        raise ValueError(f"label_col '{args.label_col}' no existe en el dataset.")

    obs_lt = _load_cols(args.obs_lt_cols_file)
    obs_u = _load_cols(args.obs_u_cols_file)
    obs_i = _load_cols(args.obs_i_cols_file)

    # filtrar columnas inexistentes
    obs_lt = [c for c in obs_lt if c in df.columns]
    obs_u = [c for c in obs_u if c in df.columns]
    obs_i = [c for c in obs_i if c in df.columns]

    y = df[args.label_col]

    def _process_block(cols: List[str], block_name: str) -> List[str]:
        if not cols:
            return []
        cat_cols = _categorical_cols(df, cols, args.cat_max_unique)
        X, mapping = _build_design(df, cols, cat_cols)
        # drop zero-variance columns
        zero_var = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        if zero_var:
            X = X.drop(columns=zero_var)

        drop_label = _drop_by_label_corr(X, y, mapping, args.label_corr_thresh)
        cols_after_label = [c for c in cols if c not in drop_label]

        # recompute X after label drop for feature correlation
        X2, mapping2 = _build_design(df, cols_after_label, _categorical_cols(df, cols_after_label, args.cat_max_unique))
        drop_feat = _drop_by_feature_corr(X2, mapping2, args.corr_thresh)
        cols_final = [c for c in cols_after_label if c not in drop_feat]

        if args.report:
            print(f"[{block_name}] cols_in={len(cols)} drop_label={len(drop_label)} drop_feat={len(drop_feat)} out={len(cols_final)}")
        return cols_final

    lt_out = _process_block(obs_lt, "OBS_LT")
    u_out = _process_block(obs_u, "OBS_U")
    i_out = _process_block(obs_i, "OBS_I")

    out_dir = Path(args.out_dir)
    _write_cols(lt_out, out_dir / "obs_lt.txt")
    _write_cols(u_out, out_dir / "obs_u.txt")
    _write_cols(i_out, out_dir / "obs_i.txt")

    if args.report:
        print(f"[OK] Guardado en: {out_dir}")


if __name__ == "__main__":
    main()
