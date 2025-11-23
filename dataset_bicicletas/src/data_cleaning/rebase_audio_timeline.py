#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import numpy as np


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if path.suffix in {".csv", ".tsv"}:
        return pd.read_csv(path)
    raise ValueError(f"Formato no soportado: {path}")


def to_seconds(series: pd.Series) -> pd.Series:
    """Convierte timestamps a segundos desde el inicio del participante."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return (series - series.min()).dt.total_seconds()
    try:
        return series.astype(float)
    except Exception:
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().all():
            return (parsed - parsed.min()).dt.total_seconds()
        raise ValueError("No se pudo convertir timestamps a segundos.")


def rebase_participant(
    df: pd.DataFrame,
    participant: str,
    session_col: str,
    start_col: str,
    timestamp_col: str,
    window_seconds: float,
    new_start_col: str,
    new_timestamp_col: str,
) -> pd.DataFrame:
    part_df = df[df["__participant_key__"] == participant].copy()
    sessions = sorted(part_df[session_col].fillna(0).astype(int).unique())

    cumulative_offset = 0.0
    ts_offset = 0.0
    rebased_starts: List[float] = []
    rebased_ts: List[float] = []

    for s in sessions:
        sess_df = part_df[part_df[session_col].fillna(0).astype(int) == s].copy()
        sess_df = sess_df.sort_values(by=timestamp_col)
        starts = sess_df[start_col].astype(float).to_numpy()
        ts_seconds = to_seconds(sess_df[timestamp_col])

        # Rebase dentro de la sesión
        base_start = starts.min()
        base_ts = ts_seconds.min()
        rebased_sess_starts = (starts - base_start) + cumulative_offset
        rebased_sess_ts = (ts_seconds - base_ts) + ts_offset

        rebased_starts.extend(rebased_sess_starts.tolist())
        rebased_ts.extend(rebased_sess_ts.tolist())

        # Actualizar offsets acumulados: asumimos que la sesión dura hasta el último start + window
        cumulative_offset = rebased_sess_starts.max() + window_seconds
        ts_offset = rebased_sess_ts.max() + window_seconds

    part_df[new_start_col] = rebased_starts
    part_df[new_timestamp_col] = rebased_ts
    return part_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebasa la línea de tiempo de audio para participantes con múltiples sesiones, "
            "eliminando huecos grandes entre sesiones. Útil para que audio_segment_start y timestamp "
            "sean continuos dentro del audio concatenado."
        )
    )
    parser.add_argument("--audio-pickle", required=True, help="Pickle/CSV con las ventanas de audio (X_vid_aud*.pkl).")
    parser.add_argument("--participant-col", default="participant")
    parser.add_argument("--session-col", default="session_id")
    parser.add_argument("--start-col", default="audio_segment_start")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument(
        "--participants",
        nargs="+",
        required=True,
        help="Participantes a rebasar (ej: P21).",
    )
    parser.add_argument("--out-pickle", default="data/processed/X_vid_aud_rebased.pkl")
    parser.add_argument("--new-start-col", default="audio_segment_start_rebased")
    parser.add_argument("--new-timestamp-col", default="timestamp_rebased")
    parser.add_argument("--overwrite-cols", action="store_true", help="Sobrescribe start/timestamp originales en vez de crear columnas nuevas.")
    args = parser.parse_args()

    df = load_table(Path(args.audio_pickle))
    required = {args.participant_col, args.session_col, args.start_col, args.timestamp_col}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Faltan columnas en audio-pickle: {missing}")

    df["__participant_key__"] = df[args.participant_col].astype(str)
    parts = set(df["__participant_key__"])
    unknown = [p for p in args.participants if p not in parts]
    if unknown:
        print(f"[warn] Participantes no encontrados en el pickle y se omitirán: {unknown}")

    rebased_chunks: List[pd.DataFrame] = []
    for p in args.participants:
        if p not in parts:
            continue
        chunk = rebase_participant(
            df=df,
            participant=p,
            session_col=args.session_col,
            start_col=args.start_col,
            timestamp_col=args.timestamp_col,
            window_seconds=args.window_seconds,
            new_start_col=args.new_start_col,
            new_timestamp_col=args.new_timestamp_col,
        )
        rebased_chunks.append(chunk)
        print(f"[rebase] {p}: {len(chunk)} filas procesadas.")

    # Unir de vuelta
    rebased_idx = pd.concat(rebased_chunks).index if rebased_chunks else []
    df_out = df.copy()
    if args.overwrite_cols:
        df_out.loc[rebased_idx, args.start_col] = pd.concat(rebased_chunks)[args.new_start_col].values
        df_out.loc[rebased_idx, args.timestamp_col] = pd.concat(rebased_chunks)[args.new_timestamp_col].values
    else:
        df_out.loc[rebased_idx, args.new_start_col] = pd.concat(rebased_chunks)[args.new_start_col].values
        df_out.loc[rebased_idx, args.new_timestamp_col] = pd.concat(rebased_chunks)[args.new_timestamp_col].values

    df_out.drop(columns="__participant_key__", inplace=True)
    out_path = Path(args.out_pickle)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_pickle(out_path)
    print(f"Pickle guardado en {out_path}")


if __name__ == "__main__":
    main()
