from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_features(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"No existe features-file: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description="Construye un dataset multimodal (tabular + video + audio) alineado y con columnas filtradas.")
    ap.add_argument("--tab-csv", type=str, default="data/processed/dataset_bicicletas_clean.csv", help="CSV tabular limpio.")
    ap.add_argument("--va-pkl", type=str, default="data/processed/X_vid_aud_patched.pkl", help="Pickle con video/audio (tensor_path_id, audio_segment_start).")
    ap.add_argument("--features-file", type=str, default="utils/feature_sets/exp1.json", help="Archivo de features tabulares.")
    ap.add_argument("--label-col", type=str, default="action_proc", help="Columna de etiqueta (numérica).")
    ap.add_argument("--timestamp-col", type=str, default="timestamp", help="Columna de join.")
    ap.add_argument("--participant-col", type=str, default="participant")
    ap.add_argument("--window-col", type=str, default="tensor_path_id", help="Nombre de la ventana/tensor de video.")
    ap.add_argument("--audio-start-col", type=str, default="audio_segment_start", help="Columna con el inicio (s) del segmento de audio.")
    ap.add_argument("--audio-root", type=str, default="/mnt/otra_particion/home/israel_gpu_data/audio_data_raw/audio_participantes_validos")
    ap.add_argument("--audio-template", type=str, default="raw_audio_{participant}.wav")
    ap.add_argument("--video-root", type=str, default="/mnt/otra_particion/home/israel_gpu_data/video_tensors")
    ap.add_argument("--keep-cols", nargs="*", default=None, help="Cols extra a preservar además de features/label/paths.")
    ap.add_argument("--out", type=str, default="data/processed/multimodal_av_join.pkl")
    args = ap.parse_args()

    tab_path = Path(args.tab_csv)
    va_path = Path(args.va_pkl)
    feats_path = Path(args.features_file)
    out_path = Path(args.out)

    if not tab_path.exists():
        raise FileNotFoundError(f"No existe tabular CSV: {tab_path}")
    if not va_path.exists():
        raise FileNotFoundError(f"No existe pickle VA: {va_path}")

    features = load_features(feats_path)
    print(f"Features cargadas ({len(features)}): {features}")

    df_tab = pd.read_csv(tab_path)
    df_va = pd.read_pickle(va_path)

    if args.timestamp_col not in df_tab.columns or args.timestamp_col not in df_va.columns:
        raise KeyError(f"Falta columna timestamp '{args.timestamp_col}' en tabular o VA.")
    if args.label_col not in df_tab.columns:
        raise KeyError(f"Falta label '{args.label_col}' en tabular.")
    if args.window_col not in df_va.columns:
        raise KeyError(f"Falta columna de ventana '{args.window_col}' en VA.")
    if args.audio_start_col not in df_va.columns:
        raise KeyError(f"Falta columna audio_start '{args.audio_start_col}' en VA.")
    if args.participant_col not in df_tab.columns or args.participant_col not in df_va.columns:
        raise KeyError(f"Falta participant '{args.participant_col}' en tabular o VA.")

    # Join por timestamp (inner para mantener consistencia)
    before_tab = len(df_tab)
    before_va = len(df_va)
    merged = pd.merge(
        df_va,
        df_tab[[args.timestamp_col, args.participant_col, args.label_col] + [c for c in features if c in df_tab.columns]],
        on=args.timestamp_col,
        how="inner",
        suffixes=("_va", "_tab"),
    )
    print(f"Tab rows={before_tab}, VA rows={before_va}, merged={len(merged)}")
    if len(merged) == 0:
        raise SystemExit("Merge vacío: revisa timestamp/archivos.")

    # Rutas absolutas
    merged["audio_route"] = merged[args.participant_col].apply(
        lambda p: f"{args.audio_root}/{args.audio_template.format(participant=p)}"
    )
    merged["frames_route"] = merged[args.window_col].apply(
        lambda w: f"{args.video_root}/{w}"
    )

    # Preservar columnas solicitadas
    keep_extra = args.keep_cols or []
    base_cols = set(features) | {args.label_col, args.timestamp_col, args.participant_col, args.window_col, args.audio_start_col, "audio_route", "frames_route"}
    base_cols |= set(keep_extra)
    cols_present = [c for c in merged.columns if c in base_cols]
    dropped = [c for c in merged.columns if c not in base_cols]

    merged = merged[cols_present].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_pickle(out_path)
    print(f"Guardado: {out_path} | cols={len(cols_present)}")
    if dropped:
        print(f"Columnas eliminadas ({len(dropped)}): {dropped}")


if __name__ == "__main__":
    main()

