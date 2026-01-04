from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def parse_run_index(idx_path: Path) -> Dict[str, dict]:
    """Parse results/run_index.txt into a dict keyed by run hash."""
    if not idx_path.exists():
        return {}

    text = idx_path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("-----") if b.strip()]
    runs: Dict[str, dict] = {}

    for block in blocks:
        header = block.splitlines()[0] if block.splitlines() else ""
        m = re.search(r"\[(?P<ts>[^\]]+)\]\s*hash=(?P<hash>\w+)\s*model=(?P<model>.+)", header)
        if not m:
            continue
        run_hash = m.group("hash")
        ts = m.group("ts").strip()
        model_name = m.group("model").strip()

        cmd_match = re.search(r"cmd:\s*(.+?)(?:\n\s*\n|$)", block, flags=re.S)
        cmd = cmd_match.group(1).strip() if cmd_match else None

        cfg_match = re.search(r"config:\s*(\{.*\})", block, flags=re.S)
        cfg = None
        if cfg_match:
            cfg_text = cfg_match.group(1).strip()
            try:
                cfg = json.loads(cfg_text)
            except Exception:
                cfg = None

        runs[run_hash] = {
            "timestamp": ts,
            "model": model_name,
            "cmd": cmd,
            "config": cfg,
            "raw": block,
        }
    return runs


def parse_eval_report_name(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Extract model and hash from an eval_report filename."""
    stem = path.stem  # already without .txt
    m = re.match(r"(?P<model>.+)-eval_report-(?P<hash>[0-9a-fA-F]+)$", stem)
    if not m:
        return None, None
    return m.group("model"), m.group("hash")


def infer_dataset(config: Optional[dict], cmd: Optional[str]) -> Optional[str]:
    """Guess dataset name (stem) from config or CLI."""
    keys = ("pickle", "pkl", "csv", "dataset", "data_path")
    if isinstance(config, dict):
        for key in keys:
            val = config.get(key)
            if val:
                return Path(str(val)).stem
    if cmd:
        m = re.search(r"--(?:pickle|pkl|csv)\s+([^\s]+)", cmd)
        if m:
            raw = m.group(1).strip().strip("\"'")
            return Path(raw).stem
    return None


def _matches_any(text: str, needles: Iterable[str]) -> bool:
    return any(n in text for n in needles)


def infer_architecture(model: Optional[str], cmd: Optional[str], config: Optional[dict]) -> str:
    """Derive architecture label from config/CLI/model name."""
    if isinstance(config, dict):
        for key in ("arch", "architecture"):
            if key in config and config[key]:
                return str(config[key])

    text = " ".join(
        [
            model or "",
            cmd or "",
            json.dumps(config, ensure_ascii=False) if isinstance(config, dict) else "",
        ]
    ).lower()

    ordered_hints = [
        ("mm_vae_audio_interpretable", ["interpretable"]),
        ("mm_vae_audio_crossatt", ["crossatt", "cross_att"]),
        ("mm_vae_audio", ["mm_vae_audio"]),
        ("mm_vae", ["mm_vae", "multimodal_vae"]),
        ("wav2vec", ["wav2vec", "w2v"]),
        ("tcn", ["tcn"]),
        ("cnn", ["cnn"]),
        ("cnn_lstm", ["cnn_lstm", "cnnlstm", "cnn-lstm"]),
        ("frame_lstm", ["frame_lstm", "frame-lstm", "clip"]),
        ("icl_v", ["iclv", "icl_v"]),
    ]
    for arch, hints in ordered_hints:
        if _matches_any(text, hints):
            return arch
    return "unknown"


def infer_modalities(model: Optional[str], cmd: Optional[str], config: Optional[dict], arch: str) -> Tuple[List[str], bool]:
    """Infer modalities involved and whether the run is multimodal."""
    text_parts = [
        model or "",
        cmd or "",
        arch or "",
        json.dumps(config, ensure_ascii=False) if isinstance(config, dict) else "",
    ]
    text = " ".join(text_parts).lower()

    mods = set()
    # Config-driven hints
    if isinstance(config, dict):
        if any(k in config for k in ("audio_col", "audio_cached_col", "audio_root", "sample_rate")):
            mods.add("audio")
        if any(k in config for k in ("path_col", "video_root", "frames_route")):
            mods.add("video")
        if config.get("features") is not None or "features_file" in config or "csv" in config or "pickle" in config:
            mods.add("tabular")
    # Textual hints (fallback)
    if _matches_any(text, ["audio", "wav2vec", "mel", "spectrogram"]):
        mods.add("audio")
    if _matches_any(text, ["video", "frame", "clip"]):
        mods.add("video")
    if _matches_any(text, ["feature", "tabular", "mnlogit", "scaler", "csv"]):
        mods.add("tabular")

    is_multimodal = len(mods) > 1 or "multimodal" in text or "mm_vae" in text
    if not mods:
        mods.add("unknown")
    return sorted(mods), is_multimodal


def collect_eval_reports(results_dir: Path) -> pd.DataFrame:
    idx_path = results_dir / "run_index.txt"
    run_index = parse_run_index(idx_path)
    eval_paths = sorted(results_dir.rglob("*eval_report*.txt"))

    rows = []
    for rpt in eval_paths:
        model_from_name, run_hash = parse_eval_report_name(rpt)
        info = run_index.get(run_hash, {})
        model = info.get("model") or model_from_name
        cmd = info.get("cmd")
        config = info.get("config")
        dataset = infer_dataset(config, cmd)
        arch = infer_architecture(model, cmd, config)
        modalities, is_multimodal = infer_modalities(model, cmd, config, arch)
        rows.append(
            {
                "model": model,
                "run_hash": run_hash,
                "arch": arch,
                "modalities": "+".join(modalities),
                "multimodal": is_multimodal,
                "dataset": dataset,
                "report_path": str(rpt),
                "timestamp": info.get("timestamp"),
                "cmd": cmd,
            }
        )

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Recopila todos los eval_report* de results y genera un DataFrame con arquitectura, modalidades y dataset."
    )
    ap.add_argument("--results-dir", type=str, default="results", help="Directorio donde viven los artefactos.")
    ap.add_argument("--save-csv", type=str, default=None, help="Ruta opcional para guardar el resumen en CSV.")
    ap.add_argument(
        "--save-datasets",
        type=str,
        default=None,
        help="Ruta opcional para guardar la lista de datasets detectados (uno por linea).",
    )
    ap.add_argument("--no-print", action="store_true", help="No imprimir el DataFrame en consola.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    df = collect_eval_reports(results_dir)

    if not args.no_print:
        if df.empty:
            print("No se encontraron eval_report en", results_dir)
        else:
            try:
                print(df.to_markdown(index=False))
            except Exception:
                print(df)

    if args.save_csv:
        out_csv = Path(args.save_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    if args.save_datasets:
        datasets = sorted({d for d in df["dataset"].tolist() if isinstance(d, str) and d})
        out_ds = Path(args.save_datasets)
        out_ds.parent.mkdir(parents=True, exist_ok=True)
        out_ds.write_text("\n".join(datasets), encoding="utf-8")
        if not args.no_print:
            print(f"Datasets detectados guardados en {out_ds}")


if __name__ == "__main__":
    main()
