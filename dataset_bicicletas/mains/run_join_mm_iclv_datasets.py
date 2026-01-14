from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.features import load_features_file


def _as_str(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str)


def main() -> None:
    ap = argparse.ArgumentParser(description="Join multimodal pickle with all_data.csv and add missing ICLV columns.")
    ap.add_argument("--all-data-csv", type=str, default="data/raw/all_data.csv")
    ap.add_argument("--multimodal-pkl", type=str, default="data/processed/multimodal_av_join_audio_cached.pkl")
    ap.add_argument("--out-pkl", type=str, default="data/processed/multimodal_av_join_audio_with_iclv.pkl")
    ap.add_argument("--participant-col", type=str, default="participant")
    ap.add_argument("--timestamp-col", type=str, default="timestamp")
    ap.add_argument("--obs-lt-cols-file", type=str, default="utils/feature_sets/obs_lt_mm.txt")
    ap.add_argument("--obs-u-cols-file", type=str, default="utils/feature_sets/obs_u_mm.txt")
    ap.add_argument("--indicator-cols-file", type=str, default="utils/feature_sets/obs_i_mm.txt")
    ap.add_argument("--impute-mode", action="store_true", help="Imputa NaN en columnas nuevas con la moda")
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

    join_cols = [args.participant_col, args.timestamp_col]
    _as_str(df_all, join_cols)
    _as_str(df_mm, join_cols)

    if df_all.duplicated(subset=join_cols).any():
        dup_count = int(df_all.duplicated(subset=join_cols).sum())
        print(f"[WARN] all_data tiene {dup_count} filas duplicadas en llaves; se usa la primera ocurrencia.")
        df_all = df_all.drop_duplicates(subset=join_cols, keep="first")

    missing_cols = [c for c in need_cols if c not in df_mm.columns]
    if not missing_cols:
        print("[OK] No hay columnas faltantes en el pickle multimodal.")
        merged = df_mm
    else:
        keep_cols = [c for c in missing_cols if c in df_all.columns]
        all_subset = df_all[join_cols + keep_cols].copy()
        merged = df_mm.merge(all_subset, on=join_cols, how="left")

        if args.impute_mode and keep_cols:
            for col in keep_cols:
                if col in merged.columns and merged[col].isna().any():
                    modes = merged[col].mode(dropna=True)
                    if len(modes) > 0:
                        merged[col] = merged[col].fillna(modes.iloc[0])

        inserted = [c for c in missing_cols if c in merged.columns]
        print(f"[OK] Columnas insertadas: {inserted}")
        still_missing = [c for c in missing_cols if c not in merged.columns]
        if still_missing:
            print(f"[WARN] Columnas faltantes tras el join: {still_missing}")

    out_path = Path(args.out_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_pickle(out_path)
    print(f"[OK] Guardado dataset mergeado en: {out_path}")
    print(f"Filas: {len(merged)} | Cols: {len(merged.columns)}")


if __name__ == "__main__":
    main()
