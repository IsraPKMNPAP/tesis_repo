"""
Evalúa modelos simples (árbol de decisión y regresión logística) sobre el dataset ancho de EEG.

Entrada:
  - data/EDA/eda_results_tabular/eeg_band_features_wide.csv

Salidas:
  - Imagen del árbol de decisión: data/EDA/eda_results_img/decision_tree_depth3.png
  - Métricas (AUC) impresas en consola.

Uso (desde dataset_neuma):
  python -m dataset_neuma.mains.run_baselines \
    --input ./data/EDA/eda_results_tabular/eeg_band_features_wide.csv \
    --out-dir ./data/EDA/eda_results_img
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Permite ejecución desde carpeta dataset_neuma
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))


def load_data(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if "bought" not in df.columns:
        raise SystemExit("La columna 'bought' no está en el dataset.")
    df["bought"] = df["bought"].fillna(-1).astype(int)
    y = df["bought"]
    feat_cols = [c for c in df.columns if c not in ["subject", "page", "product_id", "bought"]]
    X = df[feat_cols].fillna(0.0).astype(float)
    return X, y


def train_tree(X: pd.DataFrame, y: pd.Series, out_path: Path) -> float:
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)
    prob = tree.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    print("\n== Decision Tree (max_depth=3) ==")
    print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, tree.predict(X_test), digits=3))

    # Plot tree
    plt.figure(figsize=(16, 9))
    plot_tree(tree, filled=True, max_depth=3, fontsize=6, feature_names=list(X.columns), class_names=["0", "1"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return auc


def train_logreg(X: pd.DataFrame, y: pd.Series) -> float:
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, class_weight="balanced")),
        ]
    )
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    print("\n== Logistic Regression ==")
    print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, pipe.predict(X_test), digits=3))
    return auc


def main() -> None:
    parser = argparse.ArgumentParser(description="Modelos base (Decision Tree y Logistic) sobre EEG ancho.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./data/EDA/eda_results_tabular/eeg_band_features_wide.csv"),
        help="CSV ancho con features.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./data/EDA/eda_results_img"),
        help="Directorio para guardar la imagen del árbol.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"No se encontró {args.input}")

    X, y = load_data(args.input)

    tree_auc = train_tree(X, y, out_path=args.out_dir / "decision_tree_depth3.png")
    log_auc = train_logreg(X, y)

    print("\nResumen AUC:")
    print(f"  Decision Tree: {tree_auc:.4f}")
    print(f"  Logistic Reg.: {log_auc:.4f}")


if __name__ == "__main__":
    main()
