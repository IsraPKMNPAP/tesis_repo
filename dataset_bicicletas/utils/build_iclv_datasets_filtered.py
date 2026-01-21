from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _read_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_pickle(path)


def _load_cols(path: str | None) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    return [c.strip() for c in p.read_text(encoding="utf-8").splitlines() if c.strip()]


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
    dropped_orig = set()
    for orig, cols in mapping.items():
        if any(c in dropped for c in cols):
            dropped_orig.add(orig)
    return sorted(dropped_orig)


def _filter_block(df: pd.DataFrame, cols: List[str], y: pd.Series, cat_max: int, corr_thresh: float, label_corr: float) -> List[str]:
    if not cols:
        return []
    cat_cols = _categorical_cols(df, cols, cat_max)
    X, mapping = _build_design(df, cols, cat_cols)
    zero_var = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if zero_var:
        X = X.drop(columns=zero_var)
    drop_label = _drop_by_label_corr(X, y, mapping, label_corr)
    cols_after = [c for c in cols if c not in drop_label]
    X2, mapping2 = _build_design(df, cols_after, _categorical_cols(df, cols_after, cat_max))
    drop_feat = _drop_by_feature_corr(X2, mapping2, corr_thresh)
    cols_final = [c for c in cols_after if c not in drop_feat]
    return cols_final


def _expand_onehot(df: pd.DataFrame, cols: List[str], cat_max: int) -> pd.DataFrame:
    cat_cols = _categorical_cols(df, cols, cat_max)
    base = df[cols].copy()
    base = base.replace([np.inf, -np.inf], np.nan)
    base = base.fillna(base.median(numeric_only=True))
    if not cat_cols:
        return base
    dummies = pd.get_dummies(base[cat_cols].astype(str), drop_first=True)
    base_num = base.drop(columns=cat_cols, errors="ignore")
    return pd.concat([base_num, dummies], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Construye datasets ICLV/MM-ICLV filtrados por correlacion.")
    ap.add_argument("--all-data", type=str, default="data/raw/all_data.csv")
    ap.add_argument("--multimodal-pkl", type=str, default="data/processed/multimodal_av_join_audio_cached.pkl")
    ap.add_argument("--label-col", type=str, default="action_proc")
    ap.add_argument("--action-col", type=str, default="action")
    ap.add_argument("--participant-col", type=str, default="participant")
    ap.add_argument("--timestamp-col", type=str, default="timestamp")
    ap.add_argument("--frames-col", type=str, default="frames_route")
    ap.add_argument("--audio-col", type=str, default="audio_cached_path")
    ap.add_argument("--obs-lt", type=str, default="utils/feature_sets/obs_lt.txt")
    ap.add_argument("--obs-u", type=str, default="utils/feature_sets/obs_u.txt")
    ap.add_argument("--obs-i", type=str, default="utils/feature_sets/obs_i.txt")
    ap.add_argument("--obs-lt-mm", type=str, default="utils/feature_sets/obs_lt_mm.txt")
    ap.add_argument("--obs-u-mm", type=str, default="utils/feature_sets/obs_u_mm.txt")
    ap.add_argument("--obs-i-mm", type=str, default="utils/feature_sets/obs_i_mm.txt")
    ap.add_argument("--corr-thresh", type=float, default=0.6)
    ap.add_argument("--label-corr-thresh", type=float, default=0.7)
    ap.add_argument("--cat-max-unique", type=int, default=10)
    ap.add_argument("--out-iclv", type=str, default="data/processed/iclv_dataset_filtered.pkl")
    ap.add_argument("--out-mm", type=str, default="data/processed/mm_iclv_dataset_filtered.pkl")
    args = ap.parse_args()

    df_all = _read_df(Path(args.all_data)).reset_index(drop=True)
    df_mm = _read_df(Path(args.multimodal_pkl)).reset_index(drop=True)

    if args.label_col not in df_all.columns:
        if args.action_col in df_all.columns:
            label_map = {
                "accelerate": 0,
                "brake": 1,
                "decelerate": 2,
                "maintain speed": 3,
                "wait": 4,
            }
            df_all[args.label_col] = df_all[args.action_col].map(label_map)
        else:
            raise ValueError(f"No se encontro label_col '{args.label_col}' ni action_col '{args.action_col}'.")

    # join multimodal for paths
    join_cols = [c for c in [args.participant_col, args.timestamp_col] if c in df_all.columns and c in df_mm.columns]
    if join_cols:
        keep_cols = [c for c in [args.frames_col, args.audio_col] if c in df_mm.columns]
        for col in join_cols:
            df_all[col] = df_all[col].astype(str)
            df_mm[col] = df_mm[col].astype(str)
        df_mm_sub = df_mm[join_cols + keep_cols].drop_duplicates(subset=join_cols)
        df_all = df_all.merge(df_mm_sub, on=join_cols, how="left")

    # ICLV normal
    lt_cols = [c for c in _load_cols(args.obs_lt) if c in df_all.columns]
    u_cols = [c for c in _load_cols(args.obs_u) if c in df_all.columns]
    i_cols = [c for c in _load_cols(args.obs_i) if c in df_all.columns]
    base_cols = [args.label_col, args.participant_col]

    y = df_all[args.label_col]
    lt_f = _filter_block(df_all, lt_cols, y, args.cat_max_unique, args.corr_thresh, args.label_corr_thresh)
    u_f = _filter_block(df_all, u_cols, y, args.cat_max_unique, args.corr_thresh, args.label_corr_thresh)
    i_f = _filter_block(df_all, i_cols, y, args.cat_max_unique, args.corr_thresh, args.label_corr_thresh)

    X_lt = _expand_onehot(df_all, lt_f, args.cat_max_unique)
    X_u = _expand_onehot(df_all, u_f, args.cat_max_unique)
    X_i = _expand_onehot(df_all, i_f, args.cat_max_unique)
    df_iclv = pd.concat([df_all[base_cols], X_lt, X_u, X_i], axis=1)
    missing_iclv = int(df_iclv.isna().sum().sum())
    print(f"[ICLV] missing before drop: {missing_iclv}")
    df_iclv = df_iclv.dropna().reset_index(drop=True)
    print(f"[ICLV] shape={df_iclv.shape} kept lt/u/i={len(lt_f)}/{len(u_f)}/{len(i_f)}")
    Path(args.out_iclv).parent.mkdir(parents=True, exist_ok=True)
    df_iclv.to_pickle(args.out_iclv)
    print(f"[ICLV] saved: {args.out_iclv}")

    # MM ICLV
    lt_mm = [c for c in _load_cols(args.obs_lt_mm) if c in df_all.columns]
    u_mm = [c for c in _load_cols(args.obs_u_mm) if c in df_all.columns]
    i_mm = [c for c in _load_cols(args.obs_i_mm) if c in df_all.columns]
    mm_cols = [args.label_col, args.participant_col]
    if args.frames_col in df_all.columns:
        mm_cols.append(args.frames_col)
    if args.audio_col in df_all.columns:
        mm_cols.append(args.audio_col)

    lt_mm_f = _filter_block(df_all, lt_mm, y, args.cat_max_unique, args.corr_thresh, args.label_corr_thresh)
    u_mm_f = _filter_block(df_all, u_mm, y, args.cat_max_unique, args.corr_thresh, args.label_corr_thresh)
    i_mm_f = _filter_block(df_all, i_mm, y, args.cat_max_unique, args.corr_thresh, args.label_corr_thresh)

    X_lt_mm = _expand_onehot(df_all, lt_mm_f, args.cat_max_unique)
    X_u_mm = _expand_onehot(df_all, u_mm_f, args.cat_max_unique)
    X_i_mm = _expand_onehot(df_all, i_mm_f, args.cat_max_unique)
    df_mm_out = pd.concat([df_all[mm_cols], X_lt_mm, X_u_mm, X_i_mm], axis=1)
    missing_mm = int(df_mm_out.isna().sum().sum())
    print(f"[MM] missing before drop: {missing_mm}")
    df_mm_out = df_mm_out.dropna().reset_index(drop=True)
    print(f"[MM] shape={df_mm_out.shape} kept lt/u/i={len(lt_mm_f)}/{len(u_mm_f)}/{len(i_mm_f)}")
    Path(args.out_mm).parent.mkdir(parents=True, exist_ok=True)
    df_mm_out.to_pickle(args.out_mm)
    print(f"[MM] saved: {args.out_mm}")


if __name__ == "__main__":
    main()
