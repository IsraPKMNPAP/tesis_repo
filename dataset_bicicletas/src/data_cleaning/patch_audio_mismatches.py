#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torchaudio


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if path.suffix in {".csv", ".tsv"}:
        return pd.read_csv(path)
    raise ValueError(f"Formato no soportado: {path}")


def compute_required_seconds(df: pd.DataFrame, start_col: str, window_seconds: float) -> float:
    return float(df[start_col].astype(float).max() + window_seconds)


def pad_audio_to_duration(src: Path, dst: Path, target_seconds: float) -> Tuple[float, float]:
    waveform, sr = torchaudio.load(str(src))
    current_seconds = waveform.size(-1) / sr
    target_frames = int(round(target_seconds * sr))
    if waveform.size(-1) >= target_frames:
        if dst != src:
            dst.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(dst), waveform, sample_rate=sr)
        return current_seconds, current_seconds
    pad_frames = target_frames - waveform.size(-1)
    padded = torch.nn.functional.pad(waveform, (0, pad_frames))
    dst.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(dst), padded, sample_rate=sr)
    return current_seconds, padded.size(-1) / sr


def map_session_id_from_video(
    audio_df: pd.DataFrame,
    video_df: pd.DataFrame,
    participant_col: str,
    timestamp_col: str,
    session_col: str,
) -> pd.Series:
    if session_col not in video_df.columns:
        raise KeyError(f"El pickle de video no tiene la columna '{session_col}'")
    video_key = video_df[[participant_col, timestamp_col, session_col]].dropna()
    video_key[participant_col] = video_key[participant_col].astype(str)
    video_key[timestamp_col] = video_key[timestamp_col].astype(str)
    mapping: Dict[Tuple[str, str], int] = {
        (row[participant_col], row[timestamp_col]): int(row[session_col]) for _, row in video_key.iterrows()
    }
    audio_part = audio_df[participant_col].astype(str)
    audio_ts = audio_df[timestamp_col].astype(str)
    session = []
    missing = 0
    for p, ts in zip(audio_part, audio_ts):
        sid = mapping.get((p, ts))
        if sid is None:
            missing += 1
            sid = 0
        session.append(sid)
    if missing:
        print(f"[warn] No se encontró session_id para {missing} filas; se asigna 0 por defecto.")
    return pd.Series(session, index=audio_df.index, name=session_col)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parcha desincronías de audio: (1) pad de .wav si falta cobertura, (2) añade session_id desde pickle de video."
    )
    parser.add_argument("--audio-pickle", required=True, help="Pickle/CSV con ventanas de audio (X_vid_aud.pkl).")
    parser.add_argument("--audio-root", required=True, help="Carpeta con raw_audio_PXX.wav.")
    parser.add_argument("--start-col", default="audio_segment_start")
    parser.add_argument("--participant-col", default="participant")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument(
        "--participants-pad",
        nargs="*",
        default=[],
        help="Lista de participantes a forzar padding. Si se omite, aplica a todos los que necesiten cobertura.",
    )
    parser.add_argument("--audio-filename-template", default="raw_audio_{participant}.wav")
    parser.add_argument("--video-pickle", default=None, help="Pickle de video con session_id para mapear.")
    parser.add_argument("--session-col", default="session_id")
    parser.add_argument("--out-pickle", default="data/processed/X_vid_aud_patched.pkl")
    parser.add_argument(
        "--out-audio-root",
        default=None,
        help="Carpeta destino para audios parchados. Si se omite, se escribe en audio-root con sufijo _patched.",
    )
    parser.add_argument("--inplace-audio", action="store_true", help="Sobrescribir el .wav original con la versión parchada.")
    args = parser.parse_args()

    audio_df = load_table(Path(args.audio_pickle))
    required_cols = {args.participant_col, args.start_col}
    if missing := required_cols - set(audio_df.columns):
        raise SystemExit(f"Faltan columnas en audio-pickle: {missing}")

    out_audio_root = Path(args.audio_root if args.out_audio_root is None else args.out_audio_root)
    audio_root = Path(args.audio_root)
    pad_targets = set(args.participants_pad) if args.participants_pad else None

    pad_reports: List[Dict[str, object]] = []
    for participant, group in audio_df.groupby(args.participant_col):
        token = str(participant)
        required_seconds = compute_required_seconds(group, args.start_col, args.window_seconds)
        src_wav = audio_root / args.audio_filename_template.format(participant=token)
        if not src_wav.exists():
            print(f"[skip] No se encontró audio para {participant}: {src_wav}")
            continue
        info = torchaudio.info(str(src_wav))
        current_seconds = info.num_frames / (info.sample_rate or 1)
        needs_pad = required_seconds > current_seconds + 1e-6
        if pad_targets is not None and token not in pad_targets:
            needs_pad = False
        if not needs_pad:
            continue
        dst = src_wav if args.inplace_audio else out_audio_root / src_wav.name.replace(".wav", "_patched.wav")
        before, after = pad_audio_to_duration(src_wav, dst, required_seconds)
        pad_reports.append(
            {
                "participant": participant,
                "source": str(src_wav),
                "target": str(dst),
                "seconds_before": before,
                "seconds_after": after,
                "required_seconds": required_seconds,
            }
        )
        print(f"[pad] {participant}: {before:.2f}s -> {after:.2f}s (req {required_seconds:.2f}s) -> {dst}")

    if args.video_pickle:
        video_df = load_table(Path(args.video_pickle))
        if args.timestamp_col not in video_df.columns:
            raise SystemExit(f"El pickle de video no tiene '{args.timestamp_col}' para mapear session_id.")
        session_series = map_session_id_from_video(
            audio_df,
            video_df,
            participant_col=args.participant_col,
            timestamp_col=args.timestamp_col,
            session_col=args.session_col,
        )
        audio_df[args.session_col] = session_series

    out_pickle = Path(args.out_pickle)
    out_pickle.parent.mkdir(parents=True, exist_ok=True)
    audio_df.to_pickle(out_pickle)

    if pad_reports:
        report_path = out_pickle.with_suffix(".pad_report.json")
        report_path.write_text(json.dumps(pad_reports, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Reporte de padding guardado en {report_path}")

    print(f"Pickle parchado guardado en {out_pickle}")


if __name__ == "__main__":
    main()
