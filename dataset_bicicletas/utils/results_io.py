from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Sequence
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def timestamp_suffix() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def default_prefix(csv_path: str | Path, label: str) -> str:
    stem = Path(csv_path).stem
    return f"{stem}_{label}_{timestamp_suffix()}"


def save_text(text: str, out_path: str | Path):
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    out_path.write_text(text, encoding="utf-8")


def save_probs(
    probs: np.ndarray,
    classes: Sequence[str] | Sequence[int],
    out_path: str | Path,
    index: Optional[pd.Index] = None,
):
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    df = pd.DataFrame(probs, columns=list(classes))
    if index is not None:
        df.index = index
    df.to_csv(out_path, index=True)


def save_model_pickle(model, out_path: str | Path):
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    joblib.dump(model, out_path)


def compute_run_hash(config: dict, argv: Sequence[str] | None = None, model: Optional[str] = None, length: int = 8) -> str:
    try:
        payload = {
            "config": config,
            "argv": list(argv) if argv is not None else [],
            "model": model or "",
        }
        data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    except Exception:
        data = (str(config) + "\n" + " ".join(argv or []) + (model or "")).encode("utf-8")
    h = hashlib.sha1(data).hexdigest()
    return h[:length]


def artifact_name(model: str, artifact: str, run_hash: str, ext: str) -> str:
    return f"{model}-{artifact}-{run_hash}.{ext}"


def register_run(results_dir: str | Path, run_hash: str, model: str, cmd: str, config: dict):
    results_dir = Path(results_dir)
    ensure_dir(results_dir)
    idx_file = results_dir / "run_index.txt"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = (
        f"-----\n[{ts}] hash={run_hash} model={model}\n\n"
        f"cmd: {cmd}\n\n"
        f"config:\n{json.dumps(config, indent=2, ensure_ascii=False)}\n\n"
    )
    with idx_file.open("a", encoding="utf-8") as f:
        f.write(block)
