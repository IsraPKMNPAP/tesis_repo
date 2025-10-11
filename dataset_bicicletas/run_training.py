from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from dataset_bicicletas.src.data_loading.load import load_all_data
from dataset_bicicletas.src.data_cleaning.cleaning import limpiar_dataset
from dataset_bicicletas.src.features.prepare import (
    features_labels,
    split_data,
    build_preprocessor,
    encode_labels,
)
from dataset_bicicletas.src.models.baseline import train_and_evaluate


DEFAULT_FEATURES = [
    "mean_scl",
    "mean_scr",
    "scr_amplitude",
    "hr",
    "sdnn",
    "temperature_5s_log_sum",
]
DEFAULT_LABEL = "action"


def main():
    parser = argparse.ArgumentParser(description="Train models for dataset_bicicletas")
    parser.add_argument("--csv", type=str, default=None, help="Optional override CSV path")
    parser.add_argument(
        "--features",
        type=str,
        nargs="*",
        default=DEFAULT_FEATURES,
        help="Features to include",
    )
    parser.add_argument("--label", type=str, default=DEFAULT_LABEL)
    parser.add_argument("--mnlogit", action="store_true", help="Also fit Statsmodels MNLogit")
    parser.add_argument("--torch", action="store_true", help="Also train Torch embedding model")
    args = parser.parse_args()

    # Load + clean
    df_raw = load_all_data(args.csv)
    df = limpiar_dataset(df_raw)

    # Prepare data
    X, y = features_labels(df, args.features, args.label)
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor, numeric, categorical = build_preprocessor(X)

    # Baseline sklearn pipeline
    pipe, report, probs = train_and_evaluate(X_train, y_train, X_test, y_test, preprocessor)
    print("\n=== Baseline Logistic (sklearn) ===")
    print(report)
    if probs is not None:
        with np.printoptions(precision=4, suppress=True):
            print("Probabilidades primeras 5 observaciones:\n", probs[:5])

    if args.mnlogit:
        # Lazy import to avoid dependency when not requested
        from dataset_bicicletas.src.models.econ import fit_mnlogit
        X_train_proc = preprocessor.fit_transform(X_train)
        le, y_train_enc, _ = encode_labels(y_train, y_test)
        print("\n=== MNLogit (statsmodels) ===")
        res = fit_mnlogit(X_train_proc, y_train_enc)
        print(res.summary())

    # Optional: Torch embedding-style linear model
    if args.torch:
        # Lazy import to avoid dependency when not requested
        import torch
        from dataset_bicicletas.src.models.embeddings import train_simple
        X_train_proc = preprocessor.fit_transform(X_train)
        le, y_train_enc, _ = encode_labels(y_train, y_test)
        print("\n=== Torch Arkoudi-style model ===")
        X_train_tensor = torch.tensor(X_train_proc.astype(np.float32), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_enc.astype(np.int64), dtype=torch.long)
        model, hist = train_simple(
            X_train_tensor=X_train_tensor,
            y_train_tensor=y_train_tensor,
            input_dim=X_train_tensor.shape[1],
            num_classes=len(np.unique(y_train_enc)),
            lr=0.01,
            epochs=150,
        )
        print(f"Entrenado por {hist.epochs} épocas. Última pérdida={hist.losses[-1]:.4f}, Acc={hist.accs[-1]:.3f}")


if __name__ == "__main__":
    main()
