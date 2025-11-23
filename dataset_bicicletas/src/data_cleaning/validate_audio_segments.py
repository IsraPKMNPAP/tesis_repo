#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torchaudio


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if path.suffix in {".csv", ".tsv"}:
        return pd.read_csv(path)
    raise ValueError(f"Formato no soportado: {path}")


def format_participant(value, prefix: str, zero_pad: int) -> str:
    raw = str(value).strip()
    if not raw:
        raise ValueError("Participant vacío")
    raw_upper = raw.upper()
    if raw_upper.startswith(prefix.upper()):
        digits = "".join(ch for ch in raw_upper if ch.isdigit())
        if digits:
            return f"{prefix}{int(digits):0{zero_pad}d}"
        return raw_upper
    digits = re.findall(r"\d+", raw_upper)
    if digits:
        return f"{prefix}{int(digits[-1]):0{zero_pad}d}"
    raise ValueError(f"No se pudo normalizar participante: {value}")


def resolve_audio_path(root: Path, templates: List[str], participant_token: str) -> Path:
    for tmpl in templates:
        path = root / tmpl.format(participant=participant_token)
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No existe el audio para {participant_token}. Probados: "
        + ", ".join(str(root / tmpl.format(participant=participant_token)) for tmpl in templates)
    )


def audio_metadata(path: Path) -> Dict[str, float]:
    info = torchaudio.info(str(path))
    sr = info.sample_rate or 1
    duration = info.num_frames / sr
    return {
        "sample_rate": sr,
        "num_frames": info.num_frames,
        "duration_seconds": duration,
    }


def check_steps(values: pd.Series, expected: float, tol: float) -> Tuple[bool, List[float]]:
    if values.empty:
        return True, []
    diffs = values.sort_values().diff().dropna().astype(float)
    if diffs.empty:
        return True, []
    off = diffs[(diffs - expected).abs() > tol]
    return off.empty, off.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida que las ventanas audio_segment_start tengan cobertura en los .wav.")
    parser.add_argument("--pickle", required=True, help="Ruta al X_vid_aud.pkl (o CSV equivalente)")
    parser.add_argument("--audio-root", required=True, help="Carpeta con los raw_audio_PXX.wav")
    parser.add_argument("--participant-col", default="participant")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--start-col", default="audio_segment_start")
    parser.add_argument("--session-col", default=None, help="Columna de sesión; los saltos entre sesiones no cuentan como violaciones.")
    parser.add_argument("--label-col", default="action_proc")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--tolerance", type=float, default=0.25)
    parser.add_argument("--filename-template", default="raw_audio_{participant}.wav")
    parser.add_argument(
        "--fallback-template",
        default=None,
        help="Plantilla alternativa a probar si no se encuentra la principal (ej: raw_audio_{participant}_patched.wav).",
    )
    parser.add_argument("--participant-prefix", default="P")
    parser.add_argument("--participant-zero-pad", type=int, default=2)
    parser.add_argument("--out-csv", default="data/processed/audio_segments_validation.csv")
    parser.add_argument("--out-json", default="data/processed/audio_segments_validation.json")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    table = load_table(Path(args.pickle))
    for col in [args.participant_col, args.start_col]:
        if col not in table.columns:
            raise SystemExit(f"Falta la columna '{col}' en el pickle/CSV")

    audio_root = Path(args.audio_root)
    metadata_rows: List[Dict[str, object]] = []
    violations: List[str] = []

    templates = [args.filename_template] + ([args.fallback_template] if args.fallback_template else [])

    for participant, group in table.groupby(args.participant_col):
        try:
            token = format_participant(participant, args.participant_prefix, args.participant_zero_pad)
        except ValueError as exc:
            violations.append(f"{participant}: {exc}")
            continue
        try:
            audio_path = resolve_audio_path(audio_root, templates, token)
            meta = audio_metadata(audio_path)
        except FileNotFoundError as exc:
            violations.append(str(exc))
            continue

        starts_all = group[args.start_col].astype(float)
        max_start = float(starts_all.max())
        required = max_start + args.window_seconds
        coverage_ok = meta["duration_seconds"] + args.tolerance >= required

        # Revisar pasos dentro de cada sesión; ignorar saltos entre sesiones
        start_ok, start_off = True, []
        ts_ok, ts_off = True, []
        ts_span = None

        sessions = [None]
        if args.session_col and args.session_col in group.columns:
            sessions = sorted(group[args.session_col].dropna().unique().tolist())
            if not sessions:
                sessions = [None]
        else:
            sessions = [None]

        for sess in sessions:
            if sess is None:
                sub = group
            else:
                sub = group[group[args.session_col] == sess]
            if sub.empty:
                continue

            starts = sub[args.start_col].astype(float).sort_values()
            s_ok, s_off = check_steps(starts, args.window_seconds, args.tolerance)
            if not s_ok:
                start_ok = False
                start_off.extend(s_off)

            if args.timestamp_col in sub.columns:
                timestamps = pd.to_datetime(sub[args.timestamp_col], errors="coerce")
                if timestamps.notna().all():
                    ts_seconds = (timestamps - timestamps.min()).dt.total_seconds()
                    ts_span = float(ts_seconds.max() + args.window_seconds)
                    t_ok, t_off = check_steps(ts_seconds, args.window_seconds, args.tolerance)
                else:
                    ts_values = sub[args.timestamp_col].astype(float)
                    ts_span = float(ts_values.max() - ts_values.min() + args.window_seconds)
                    t_ok, t_off = check_steps(ts_values, args.window_seconds, args.tolerance)
                if not t_ok:
                    ts_ok = False
                    ts_off.extend(t_off)

        ok = coverage_ok and start_ok and ts_ok
        metadata_rows.append(
            {
                "participant": participant,
                "participant_token": token,
                "audio_path": str(audio_path),
                "instances": int(len(group)),
                "max_start": max_start,
                "required_seconds": required,
                "audio_seconds": meta["duration_seconds"],
                "timestamp_span_seconds": ts_span,
                "coverage_ok": coverage_ok,
                "start_step_ok": start_ok,
                "timestamp_step_ok": ts_ok,
                "start_step_violations": start_off,
                "timestamp_step_violations": ts_off,
                "ok": ok,
            }
        )
        if not ok:
            violations.append(f"{participant}: coverage={coverage_ok}, start_ok={start_ok}, ts_ok={ts_ok}")

    meta_df = pd.DataFrame(metadata_rows).sort_values("participant")
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(out_csv, index=False)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metadata_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resumen guardado en {out_csv} y {out_json}")

    if violations:
        print("Problemas detectados:")
        for msg in violations:
            print(f" - {msg}")
        if args.fail_on_error:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
