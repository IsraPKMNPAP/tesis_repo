from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Asegurar que el directorio padre (raíz del paquete) esté en sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ejecutar desde el directorio dataset_bicicletas
from src.data_loading.load import load_all_data
from src.data_cleaning.cleaning import limpiar_dataset
from src.features.prepare import (
    features_labels,
    split_data,
    build_preprocessor,
    encode_labels,
)
from src.models.baseline import train_and_evaluate
from utils.results_io import (
    default_prefix,
    ensure_dir,
    save_model_pickle,
    save_probs,
    save_text,
)
from utils.features import feature_hash


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
    parser.add_argument("--csv", type=str, required=True, help="Ruta al CSV a utilizar (requerido)")
    parser.add_argument(
        "--features",
        type=str,
        nargs="*",
        default=DEFAULT_FEATURES,
        help="Features a incluir",
    )
    parser.add_argument("--features-file", type=str, default=None, help="Archivo con lista de features (json o txt)")
    parser.add_argument("--label", type=str, default=DEFAULT_LABEL)
    parser.add_argument("--mnlogit", action="store_true", help="Ajustar también MNLogit (statsmodels)")
    parser.add_argument("--torch", action="store_true", help="Entrenar modelo Torch sencillo")
    parser.add_argument("--no-clean", action="store_true", help="No aplicar limpieza (usar CSV ya procesado)")
    parser.add_argument("--prefix", type=str, default=None, help="Prefijo para archivos en results/")
    args = parser.parse_args()

    # Carga + limpieza
    # Cargar lista de features desde archivo si se provee
    if args.features_file:
        from utils.features import load_features_file  # lazy import to avoid circulars
        loaded = load_features_file(args.features_file)
        if loaded:
            args.features = loaded

    df_raw = load_all_data(args.csv)
    df = df_raw if args.no_clean else limpiar_dataset(df_raw)

    # Preparación de datos
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

    # Guardado de resultados baseline
    # Construir prefijo si no se pasa: incluye recuento y hash corto de features
    auto_prefix = f"{default_prefix(args.csv, args.label)}_{len(args.features)}f_{feature_hash(args.features)}"
    prefix = args.prefix or auto_prefix
    results_dir = Path("results")
    ensure_dir(results_dir)
    # Guardar config de corrida
    config = {
        "csv": args.csv,
        "features": list(args.features),
        "label": args.label,
        "mnlogit": args.mnlogit,
        "torch": args.torch,
        "no_clean": args.no_clean,
        "prefix": prefix,
    }
    (results_dir / f"{prefix}_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    # Reporte
    save_text(report, results_dir / f"{prefix}_baseline_report.txt")
    # Probabilidades
    classes = getattr(pipe.named_steps["classifier"], "classes_", [])
    save_probs(
        probs=probs,
        classes=classes,
        out_path=results_dir / f"{prefix}_baseline_proba.csv",
        index=getattr(y_test, "index", None),
    )
    # Modelo (pipeline entero)
    save_model_pickle(pipe, results_dir / f"{prefix}_baseline_model.pkl")

    # MNLogit opcional
    if args.mnlogit:
        from src.models.econ import fit_mnlogit  # lazy import

        X_train_proc = preprocessor.fit_transform(X_train)
        le, y_train_enc, _ = encode_labels(y_train, y_test)
        print("\n=== MNLogit (statsmodels) ===")
        res = fit_mnlogit(X_train_proc, y_train_enc)
        summary = res.summary().as_text()
        print(summary)
        save_text(summary, results_dir / f"{prefix}_mnlogit_summary.txt")

    # Torch opcional
    if args.torch:
        import torch  # lazy import
        from src.models.embeddings import train_simple

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
        print(
            f"Entrenado por {hist.epochs} épocas. Última pérdida={hist.losses[-1]:.4f}, Acc={hist.accs[-1]:.3f}"
        )
        # Guardado del modelo e historial
        torch.save(model.state_dict(), results_dir / f"{prefix}_torch_model.pt")
        hist_df = pd.DataFrame({"loss": hist.losses, "acc": hist.accs})
        hist_df.to_csv(results_dir / f"{prefix}_torch_history.csv", index=False)


if __name__ == "__main__":
    main()
