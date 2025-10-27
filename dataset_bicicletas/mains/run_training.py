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
    ensure_dir,
    save_model_pickle,
    save_probs,
    save_text,
    compute_run_hash,
    artifact_name,
    register_run,
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

    # Guardado de resultados baseline con nuevo esquema
    results_dir = Path("results")
    ensure_dir(results_dir)
    model_name = "LogReg"
    # Guardar config de corrida
    base_config = {
        "csv": args.csv,
        "features": list(args.features),
        "label": args.label,
        "mnlogit": args.mnlogit,
        "torch": args.torch,
        "no_clean": args.no_clean,
    }
    run_hash = compute_run_hash(base_config, sys.argv, model=model_name)
    (results_dir / artifact_name(model_name, "config", run_hash, "json")).write_text(
        json.dumps(base_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Reporte
    save_text(report, results_dir / artifact_name(model_name, "eval_report", run_hash, "txt"))
    # Probabilidades
    classes = getattr(pipe.named_steps["classifier"], "classes_", [])
    save_probs(
        probs=probs,
        classes=classes,
        out_path=results_dir / artifact_name(model_name, "eval_proba", run_hash, "csv"),
        index=getattr(y_test, "index", None),
    )
    # Modelo (pipeline entero)
    save_model_pickle(pipe, results_dir / artifact_name(model_name, "model", run_hash, "pkl"))
    # Index entry
    register_run(results_dir, run_hash, model_name, cmd=" ".join(sys.argv), config=base_config)

    # MNLogit opcional
    if args.mnlogit:
        from src.models.econ import fit_mnlogit  # lazy import

        X_train_proc = preprocessor.fit_transform(X_train)
        le, y_train_enc, _ = encode_labels(y_train, y_test)
        print("\n=== MNLogit (statsmodels) ===")
        res = fit_mnlogit(X_train_proc, y_train_enc)
        summary = res.summary().as_text()
        print(summary)
        save_text(
            summary,
            results_dir / artifact_name("MNLogit", "summary", compute_run_hash(base_config, sys.argv, model="MNLogit"), "txt"),
        )

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
        torch_model_name = "TorchEmbed"
        torch_hash = compute_run_hash(base_config, sys.argv, model=torch_model_name)
        torch.save(model.state_dict(), results_dir / artifact_name(torch_model_name, "model", torch_hash, "pt"))
        hist_df = pd.DataFrame({"loss": hist.losses, "acc": hist.accs})
        hist_df.to_csv(results_dir / artifact_name(torch_model_name, "history", torch_hash, "csv"), index=False)
        register_run(results_dir, torch_hash, torch_model_name, cmd=" ".join(sys.argv), config=base_config)


if __name__ == "__main__":
    main()
