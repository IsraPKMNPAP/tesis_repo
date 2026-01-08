from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .results_io import ensure_dir, compute_run_hash, artifact_name, save_text


def mean_nll_from_logprobs(log_probs: np.ndarray, y_true: np.ndarray) -> float:
    """Calcula NLL promedio dado log-probabilidades [N, C] y etiquetas."""
    log_probs = np.asarray(log_probs)
    y_true = np.asarray(y_true, dtype=int)
    idx = (np.arange(len(y_true)), y_true)
    return float(-log_probs[idx].mean())


def classification_report_basic(y_true: np.ndarray, y_pred: np.ndarray, log_probs: Optional[np.ndarray] = None) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    out = {
        "acc": float(acc),
        "f1_macro": float(f1_macro),
    }
    for i, v in enumerate(f1_per_class):
        out[f"f1_class_{i}"] = float(v)
    if log_probs is not None:
        out["mean_nll"] = mean_nll_from_logprobs(log_probs, y_true)
    return out


def pseudo_r2_mcfadden(log_lik_model: float, y_true: np.ndarray) -> float:
    """Pseudo R^2 de McFadden con baseline de frecuencias empíricas."""
    y_true = np.asarray(y_true, dtype=int)
    counts = np.bincount(y_true)
    probs = counts / counts.sum()
    log_lik_null = float(np.sum(np.log(probs[y_true] + 1e-12)))
    if log_lik_null == 0:
        return float("nan")
    return 1.0 - (log_lik_model / log_lik_null)


def save_metrics(
    metrics: Dict[str, float],
    results_dir: Path,
    model_name: str,
    config: Dict,
    run_hash: Optional[str] = None,
) -> None:
    results_dir = Path(results_dir) / model_name
    ensure_dir(results_dir)
    if run_hash is None:
        argv = config.get("argv") if isinstance(config, dict) else None
        run_hash = compute_run_hash(config, argv, model=model_name)
    lines = [f"{k}: {v}" for k, v in sorted(metrics.items())]
    save_text("\n".join(lines), results_dir / artifact_name(model_name, "metrics", run_hash, "txt"))
    cfg_path = results_dir / artifact_name(model_name, "config", run_hash, "json")
    cfg_path.write_text(
        __import__("json").dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    cmd_path = results_dir / artifact_name(model_name, "cmd", run_hash, "txt")
    cmd_path.write_text(" ".join(config.get("argv", [])), encoding="utf-8")


def iclv_coeff_stats(hessian_res, top_k: int = 5) -> Dict[str, float]:
    """Extrae theta, se, t-stat de los primeros top_k coeficientes de un HessianResult."""
    out = {}
    for name, theta, se, tstat in zip(hessian_res.names[:top_k], hessian_res.theta[:top_k], hessian_res.std[:top_k], hessian_res.tstat[:top_k]):
        out[f"theta_{name}"] = float(theta)
        out[f"se_{name}"] = float(se)
        out[f"t_{name}"] = float(tstat)
    return out
