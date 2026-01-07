from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from utils.features import load_features_file


def main():
    ap = argparse.ArgumentParser(description="Join all_data.csv with multimodal pickle for ICLV.")
    ap.add_argument("--all-data-csv", type=str, default="data/raw/all_data.csv")
    ap.add_argument("--multimodal-pkl", type=str, default="data/processed/multimodal_av_join_audio_cached.pkl")
    ap.add_argument("--out-pkl", type=str, default="data/processed/multimodal_av_join_audio_with_iclv.pkl")
    ap.add_argument("--participant-col", type=str, default="participant")
    ap.add_argument("--timestamp-col", type=str, default="timestamp")
    ap.add_argument("--obs-lt-cols-file", type=str, default="utils/feature_sets/obs_lt.txt")
    ap.add_argument("--obs-u-cols-file", type=str, default="utils/feature_sets/obs_u.txt")
    ap.add_argument("--indicator-cols-file", type=str, default="utils/feature_sets/obs_i.txt")
    args = ap.parse_args()

    all_data_path = Path(args.all_data_csv)
    mm_path = Path(args.multimodal_pkl)
    if not all_data_path.exists():
        raise FileNotFoundError(f"No existe {all_data_path}")
    if not mm_path.exists():
        raise FileNotFoundError(f"No existe {mm_path}")

    obs_lt = load_features_file(args.obs_lt_cols_file) or []
    obs_u = load_features_file(args.obs_u_cols_file) or []
    obs_i = load_features_file(args.indicator_cols_file) or []
    need_cols = list(dict.fromkeys(obs_lt + obs_u + obs_i))

    df_all = pd.read_csv(all_data_path, low_memory=False)
    df_mm = pd.read_pickle(mm_path)

    # Normalize join keys
    for df in (df_all, df_mm):
        if args.participant_col in df.columns:
            df[args.participant_col] = df[args.participant_col].astype(str)
        if args.timestamp_col in df.columns:
            df[args.timestamp_col] = df[args.timestamp_col].astype(str)

    # Only bring required columns from all_data to avoid collisions
    keep_cols = [c for c in need_cols if c in df_all.columns and c not in df_mm.columns]
    join_cols = [args.participant_col, args.timestamp_col]
    all_subset = df_all[join_cols + keep_cols].copy()

    merged = df_mm.merge(all_subset, on=join_cols, how="left")

    missing = [c for c in need_cols if c not in merged.columns]
    if missing:
        print(f"[WARN] Columnas faltantes en el merge: {missing}")

    out_path = Path(args.out_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_pickle(out_path)
    print(f"Guardado dataset mergeado en: {out_path}")
    print(f"Filas: {len(merged)} | Cols: {len(merged.columns)}")


if __name__ == "__main__":
    main()
