from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pos_label: int = 1,
) -> Dict:
    """Devuelve acc, f1_macro, f1_pos, f1_neg, auc, nll (mean)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "acc": float(accuracy_score(y_true, y_pred)) if y_true.size else float("nan"),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true.size else float("nan"),
        "f1_pos": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)) if y_true.size else float("nan"),
        "f1_neg": float(f1_score(y_true, y_pred, pos_label=1 - pos_label, zero_division=0)) if y_true.size else float("nan"),
    }
    if y_true.size and len(np.unique(y_true)) > 1:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["auc"] = float("nan")
    else:
        metrics["auc"] = float("nan")
    try:
        metrics["nll"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        metrics["nll"] = float("nan")
    return metrics


def pseudo_r2_mcfadden(log_lik_model: float, y_true: np.ndarray, num_choices: int) -> float:
    """Pseudo-R2 de McFadden."""
    y_true = np.asarray(y_true)
    if y_true.size == 0:
        return float("nan")
    if num_choices == 2:
        p = np.clip(y_true.mean(), 1e-6, 1 - 1e-6)
        ll_null = (y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).sum()
    else:
        counts = np.bincount(y_true.astype(int), minlength=num_choices)
        probs = counts / counts.sum()
        probs = np.clip(probs, 1e-6, 1.0)
        ll_null = np.log(probs[y_true.astype(int)]).sum()
    return 1 - (log_lik_model / ll_null)


def summarize_coefs(names: Sequence[str], theta: np.ndarray, std: Optional[np.ndarray] = None, top_k: int = 5) -> Dict:
    theta = np.asarray(theta).flatten()
    std = np.asarray(std).flatten() if std is not None else None
    pairs = list(zip(names, theta)) if names else [(f"coef_{i}", v) for i, v in enumerate(theta)]
    top = pairs[:top_k]
    out = []
    for i, (n, v) in enumerate(top):
        entry = {"name": n, "coef": float(v)}
        if std is not None and i < len(std):
            entry["std"] = float(std[i])
            entry["tstat"] = float(v / (std[i] + 1e-12))
        out.append(entry)
    return {"top_coefs": out}


def save_metrics(metrics: Dict, results_dir: Path, filename: str = "metrics.json") -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
