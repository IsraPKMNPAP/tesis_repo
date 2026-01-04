from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def load_json_safe(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_metrics(metrics: Optional[dict]) -> Tuple[dict, str, str, str]:
    """Return metrics values plus status and missing/extra keys."""
    expected = {"acc@0.5", "f1@0.5", "auc", "best_thr", "best_acc", "best_f1"}
    if not isinstance(metrics, dict):
        return {}, "missing_or_invalid", "", ""
    missing = sorted(list(expected - set(metrics.keys())))
    extra = sorted(list(set(metrics.keys()) - expected))
    status = "ok" if not missing else "incomplete"
    return metrics, status, ",".join(missing), ",".join(extra)


def _matches_any(text: str, needles: Iterable[str]) -> bool:
    return any(n in text for n in needles)


def infer_dataset(meta: Optional[dict], extra: Optional[dict], path: Path) -> Optional[str]:
    """Guess dataset name from metadata/config or enclosing path."""
    sources = [meta, extra]
    keys = ("data", "data_path", "csv", "pickle", "pkl", "input_csv", "output_csv", "dataset")
    for src in sources:
        if isinstance(src, dict):
            for key in keys:
                val = src.get(key)
                if val:
                    try:
                        return Path(str(val)).stem
                    except Exception:
                        return str(val)
    # Heuristic: parent directory name if it looks like a dataset file stem
    for part in reversed(path.parts):
        if part.lower().endswith((".csv", ".pkl", ".pickle")):
            return Path(part).stem
    return None


def infer_architecture(model: str, meta: Optional[dict], extra: Optional[dict]) -> str:
    text = " ".join(
        [
            model,
            json.dumps(meta, ensure_ascii=False) if isinstance(meta, dict) else "",
            json.dumps(extra, ensure_ascii=False) if isinstance(extra, dict) else "",
        ]
    ).lower()
    ordered_hints = [
        ("multimodal_icl_v", ["multimodal_icl_v"]),
        ("icl_v", ["icl_v"]),
        ("multimodal_fusion_logits", ["fusionclassifierlogits", "fusionvaelogits", "logits_baseline"]),
        ("multimodal_fusion", ["fusionclassifier", "fusionvae", "fusion_baseline"]),
        ("tabular_clip_fusion", ["tabular_image_fusion", "tabular_clip"]),
        ("tabular_mlp", ["tabular", "mlp"]),
        ("eeg_cnn", ["eeg_cnn"]),
        ("eeg_autoencoder", ["autoencoder"]),
    ]
    for arch, hints in ordered_hints:
        if _matches_any(text, hints):
            return arch
    return "unknown"


def infer_modalities(model: str, meta: Optional[dict], extra: Optional[dict], arch: str) -> Tuple[List[str], bool]:
    text = " ".join(
        [
            model,
            arch,
            json.dumps(meta, ensure_ascii=False) if isinstance(meta, dict) else "",
            json.dumps(extra, ensure_ascii=False) if isinstance(extra, dict) else "",
        ]
    ).lower()
    mods = set()
    if _matches_any(text, ["eeg"]):
        mods.add("eeg")
    if _matches_any(text, ["clip", "image", "img"]):
        mods.add("image")
    if _matches_any(text, ["tab", "tabular", "feature"]):
        mods.add("tabular")
    if _matches_any(text, ["multimodal"]):
        mods.update(["tabular", "image"])
    if not mods:
        mods.add("unknown")
    is_multi = len(mods) > 1
    return sorted(mods), is_multi


def collect_runs(results_dir: Path) -> pd.DataFrame:
    # Consider directories that contain metrics.json, metadata.json, or eval_report-like files
    candidates: Dict[Path, Dict[str, Optional[Path]]] = {}

    for path in results_dir.rglob("metrics.json"):
        candidates.setdefault(path.parent, {})["metrics"] = path
    for path in results_dir.rglob("metadata.json"):
        candidates.setdefault(path.parent, {})["metadata"] = path
    for path in results_dir.rglob("*eval_report*.txt"):
        candidates.setdefault(path.parent, {})["eval_report"] = path

    rows = []
    for run_dir, files in sorted(candidates.items()):
        metrics_path = files.get("metrics")
        meta_path = files.get("metadata")
        eval_path = files.get("eval_report")

        metrics = load_json_safe(metrics_path) if metrics_path else None
        metadata = load_json_safe(meta_path) if meta_path else None
        metrics_vals, metrics_status, metrics_missing, metrics_extra = parse_metrics(metrics)

        # Use relative path to keep nested runs distinguishable
        try:
            model_name = str(run_dir.relative_to(results_dir))
        except Exception:
            model_name = run_dir.name
        arch = infer_architecture(model_name, metadata, metrics)
        modalities, is_multimodal = infer_modalities(model_name, metadata, metrics, arch)
        dataset = infer_dataset(metadata, metrics, run_dir)

        missing: List[str] = []
        if not metrics_path:
            missing.append("metrics.json")
        if not eval_path:
            missing.append("eval_report")
        rows.append(
            {
                "model": model_name,
                "arch": arch,
                "modalities": "+".join(modalities),
                "multimodal": is_multimodal,
                "dataset": dataset,
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path) if metrics_path else None,
                "eval_report": str(eval_path) if eval_path else None,
                "metadata_path": str(meta_path) if meta_path else None,
                "missing": ",".join(missing) if missing else "",
                "metrics_status": metrics_status,
                "metrics_missing": metrics_missing,
                "metrics_extra": metrics_extra,
                "acc@0.5": metrics_vals.get("acc@0.5"),
                "f1@0.5": metrics_vals.get("f1@0.5"),
                "auc": metrics_vals.get("auc"),
                "best_thr": metrics_vals.get("best_thr"),
                "best_acc": metrics_vals.get("best_acc"),
                "best_f1": metrics_vals.get("best_f1"),
            }
        )

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Agrega resultados de dataset_neuma buscando metrics.json/eval_report y resumiendo arquitectura y modalidades."
    )
    ap.add_argument("--results-dir", type=str, default="results", help="Directorio base de resultados.")
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
    df = collect_runs(results_dir)

    if not args.no_print:
        if df.empty:
            print(f"No se encontraron runs en {results_dir}")
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
