from __future__ import annotations

import json
from typing import Iterable, Optional
import numpy as np


def class_balance_report(y_true: Iterable[int], y_pred: Iterable[int], num_classes: Optional[int] = None):
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    if num_classes is None:
        k_true = (np.nanmax(y_true) + 1) if y_true.size else 0
        k_pred = (np.nanmax(y_pred) + 1) if y_pred.size else 0
        num_classes = int(max(k_true, k_pred))
    counts_true = np.bincount(np.clip(y_true, 0, 10**6), minlength=num_classes)
    counts_pred = np.bincount(np.clip(y_pred, 0, 10**6), minlength=num_classes)
    report = {
        "num_classes": int(num_classes),
        "support_true": counts_true.tolist(),
        "support_pred": counts_pred.tolist(),
        "only_one_pred_class": int(len(np.unique(y_pred)) == 1),
    }
    return report


def logits_stats(logits: np.ndarray, sample_rows: int = 5):
    # logits: [N, C]
    arr = np.asarray(logits, dtype=float)
    finite_mask = np.isfinite(arr)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    total = int(arr.size)
    if arr.ndim == 2:
        per_class_mean = np.nanmean(arr, axis=0).tolist()
        per_class_std = np.nanstd(arr, axis=0).tolist()
    else:
        per_class_mean = None
        per_class_std = None
    overall_mean = float(np.nanmean(arr))
    overall_std = float(np.nanstd(arr))
    finite_vals = arr[finite_mask]
    min_val = float(np.nanmin(finite_vals)) if finite_vals.size else float('nan')
    max_val = float(np.nanmax(finite_vals)) if finite_vals.size else float('nan')
    stats = {
        "shape": list(arr.shape),
        "per_class_mean": per_class_mean,
        "per_class_std": per_class_std,
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "min": min_val,
        "max": max_val,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "finite_fraction": float(finite_vals.size / max(1, total)),
        "sample": arr[:sample_rows].tolist(),
    }
    return stats


def print_debug_summary(y_true, y_pred, logits: Optional[np.ndarray] = None, num_classes: Optional[int] = None):
    rep = class_balance_report(y_true, y_pred, num_classes=num_classes)
    print("=== Class Balance (true vs pred) ===")
    print(json.dumps(rep, indent=2))
    if logits is not None:
        ls = logits_stats(logits)
        print("\n=== Logits Stats ===")
        print(json.dumps(ls, indent=2))
