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


def detect_sessions_by_gap(
    ts_seconds: pd.Series, gap_threshold: float, min_len: int = 1
) -> pd.Series:
    """Genera IDs de sesión cuando hay saltos mayores al umbral."""
    ts_sorted = ts_seconds.reset_index(drop=True)
    diffs = ts_sorted.diff().fillna(0.0)
    session_ids = [0]
    current = 0
    for d in diffs.iloc[1:]:
        if d > gap_threshold:
            current += 1
        session_ids.append(current)
    return pd.Series(session_ids, index=ts_seconds.index, name="session_detected")


def rebase_participant(
    df: pd.DataFrame,
    participant: str,
    session_col: str,
    start_col: str,
    timestamp_col: str,
    window_seconds: float,
    new_start_col: str,
    new_timestamp_col: str,
    auto_session: bool,
    gap_threshold: float,
) -> pd.DataFrame:
    part_df = df[df["__participant_key__"] == participant].copy()
    # Determinar sesiones: usar columna si existe y no es todo NaN; si no, autogenerar por gaps
    if session_col in part_df.columns and part_df[session_col].notna().any():
        sessions_series = part_df[session_col].fillna(method="ffill").fillna(method="bfill").astype(int)
    elif auto_session:
        ts_seconds = to_seconds(part_df[timestamp_col])
        sessions_series = detect_sessions_by_gap(ts_seconds, gap_threshold=gap_threshold)
    else:
        sessions_series = pd.Series(0, index=part_df.index)

    part_df["__session_id__"] = sessions_series
    sessions = sorted(part_df["__session_id__"].astype(int).unique())

    cumulative_offset = 0.0
    ts_offset = 0.0
    rebased_starts: Dict[int, List[float]] = {}
    rebased_ts: Dict[int, List[float]] = {}

    for s in sessions:
        sess_df = part_df[part_df["__session_id__"] == s].copy().sort_values(by=timestamp_col)
        starts = sess_df[start_col].astype(float).to_numpy()
        ts_seconds = to_seconds(sess_df[timestamp_col])

        base_start = starts.min()
        base_ts = ts_seconds.min()
        rebased_sess_starts = (starts - base_start) + cumulative_offset
        rebased_sess_ts = (ts_seconds - base_ts) + ts_offset

        rebased_starts[s] = rebased_sess_starts.tolist()
        rebased_ts[s] = rebased_sess_ts.tolist()

        cumulative_offset = rebased_sess_starts.max() + window_seconds
        ts_offset = rebased_sess_ts.max() + window_seconds

    # Escribir de nuevo con el mismo orden original
    part_df[new_start_col] = np.nan
    part_df[new_timestamp_col] = np.nan
    for s in sessions:
        idx = part_df[part_df["__session_id__"] == s].sort_values(by=timestamp_col).index
        part_df.loc[idx, new_start_col] = rebased_starts[s]
        part_df.loc[idx, new_timestamp_col] = rebased_ts[s]

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
    parser.add_argument("--auto-session", action="store_true", help="Detecta sesiones por huecos grandes si session_id no está disponible.")
    parser.add_argument("--gap-threshold", type=float, default=30.0, help="Umbral de segundos para detectar nueva sesión cuando auto-session está activo.")
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
            auto_session=args.auto_session,
            gap_threshold=args.gap_threshold,
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
