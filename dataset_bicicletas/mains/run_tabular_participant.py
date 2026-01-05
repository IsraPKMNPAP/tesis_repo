from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.features import load_features_file
from utils.results_io import ensure_dir
from utils.splits import split_by_participant, format_split_report
from utils.metrics_eval import classification_report_basic, save_metrics


def build_pipe(numeric, categorical):
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)])


def main():
    ap = argparse.ArgumentParser(description="Baseline tabular con split por participante (sin leakage).")
    ap.add_argument("--pkl", type=str, default="data/processed/multimodal_av_join_audio_cached.pkl")
    ap.add_argument("--features-file", type=str, default="utils/feature_sets/exp1.json")
    ap.add_argument("--label-col", type=str, default="action_proc")
    ap.add_argument("--participant-col", type=str, default="participant")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-dir", type=str, default="results")
    args = ap.parse_args()

    df = pd.read_pickle(args.pkl).reset_index(drop=True)
    feats = load_features_file(args.features_file) or []
    feats = [f for f in feats if f in df.columns]
    if not feats:
        raise ValueError("No se encontraron features válidas en el pickle.")

    df_tr, df_val, df_te, info = split_by_participant(
        df, participant_col=args.participant_col, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    print(format_split_report(info))

    # preparar columnas
    X_tr = df_tr[feats].copy()
    y_tr = df_tr[args.label_col].astype(int)
    X_val = df_val[feats].copy()
    y_val = df_val[args.label_col].astype(int)
    X_te = df_te[feats].copy()
    y_te = df_te[args.label_col].astype(int)

    numeric = X_tr.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical = X_tr.select_dtypes(exclude=["int64", "float64", "int32", "float32"]).columns.tolist()

    pipe = build_pipe(numeric, categorical)
    pipe.fit(X_tr, y_tr)

    def eval_split(X, y):
        probs = pipe.predict_proba(X)
        logp = np.log(probs + 1e-12)
        preds = probs.argmax(axis=1)
        return classification_report_basic(y, preds, log_probs=logp)

    metrics_val = eval_split(X_val, y_val)
    metrics_test = eval_split(X_te, y_te) if len(df_te) else {}

    # save
    results_dir = Path(args.results_dir)
    ensure_dir(results_dir)
    base_config = {
        "pkl": args.pkl,
        "features_file": args.features_file,
        "features": feats,
        "label_col": args.label_col,
        "participant_col": args.participant_col,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "seed": args.seed,
        "argv": sys.argv,
    }
    all_metrics = {f"val_{k}": v for k, v in metrics_val.items()}
    all_metrics.update({f"test_{k}": v for k, v in metrics_test.items()})
    save_metrics(all_metrics, results_dir, model_name="TabularBaseline", config=base_config)
    # Guardar split info
    info_path = results_dir / "TabularBaseline" / "split_info.txt"
    info_path.write_text(format_split_report(info), encoding="utf-8")
    print(f"Resultados guardados en: {results_dir / 'TabularBaseline'}")


if __name__ == "__main__":
    main()
