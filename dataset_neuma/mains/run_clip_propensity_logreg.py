"""
Regresión logística (L2) sobre embeddings CLIP para predecir alta/baja propensión de compra.

Definición de etiqueta (por producto = (page, product_id)):
  - purchase_rate = mean(bought) sobre todos los sujetos
  - y = 1 si purchase_rate >= Q3 (p75)
  - y = 0 si purchase_rate <= Q1 (p25)
  - se ignoran los productos con Q1 < purchase_rate < Q3 o NaN

Entradas:
  - data/processed/image_embeddings/embeddings_index.csv (rutas a embedding.npy)
  - data/processed/products_all_with_images.csv (para bought -> purchase_rate)

Salida:
  - solo consola: AUC + tamaño del set + umbrales

Uso (desde dataset_neuma):
  python -m dataset_neuma.mains.run_clip_propensity_logreg --embeddings-dir ./data/processed/image_embeddings --products ./data/processed/products_all_with_images.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Permite ejecución desde carpeta dataset_neuma
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))


def load_embeddings(paths: List[str]) -> np.ndarray:
    vecs = [np.load(Path(p)).astype(np.float32) for p in paths]
    return np.stack(vecs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="LogReg sobre embeddings CLIP para alta/baja propensión de compra.")
    parser.add_argument("--embeddings-dir", type=Path, default=Path("./data/processed/image_embeddings"))
    parser.add_argument("--products", type=Path, default=Path("./data/processed/products_all_with_images.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0, help="Inverso de fuerza de regularización L2.")
    args = parser.parse_args()

    index_path = args.embeddings_dir / "embeddings_index.csv"
    if not index_path.exists():
        raise SystemExit(f"No se encontró {index_path}")
    if not args.products.exists():
        raise SystemExit(f"No se encontró {args.products}")

    emb_index = pd.read_csv(index_path)
    if emb_index.empty:
        raise SystemExit("embeddings_index.csv está vacío.")
    if not {"page", "product_id", "embedding_path"}.issubset(emb_index.columns):
        raise SystemExit("embeddings_index.csv debe contener page, product_id, embedding_path")

    df_prod = pd.read_csv(args.products)
    if not {"page", "product_id", "bought"}.issubset(df_prod.columns):
        raise SystemExit("products_all_with_images.csv debe contener page, product_id, bought")

    rates = (
        df_prod.dropna(subset=["bought"])
        .groupby(["page", "product_id"])["bought"]
        .mean()
        .rename("purchase_rate")
        .reset_index()
    )

    df = emb_index.merge(rates, on=["page", "product_id"], how="left")
    df = df.dropna(subset=["purchase_rate"])

    q1 = float(df["purchase_rate"].quantile(0.25))
    q3 = float(df["purchase_rate"].quantile(0.75))

    df["y"] = np.where(df["purchase_rate"] >= q3, 1, np.where(df["purchase_rate"] <= q1, 0, np.nan))
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)

    if df["y"].nunique() < 2:
        raise SystemExit("No hay suficientes clases luego del filtrado por cuartiles.")

    # Cargar embeddings solo para los seleccionados
    X = load_embeddings(df["embedding_path"].tolist())
    y = df["y"].to_numpy()

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, C=args.C, penalty="l2", solver="lbfgs", class_weight="balanced")),
        ]
    )
    pipe.fit(X, y)

    prob = pipe.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, prob)

    print("== CLIP Embeddings -> High/Low Purchase Propensity (LogReg L2) ==")
    print(f"Q1={q1:.4f}  Q3={q3:.4f}")
    print(f"Used samples: {len(df)} (class0={int((y==0).sum())}, class1={int((y==1).sum())})")
    print(f"Seed: {args.seed}, C: {args.C}")
    print(f"AUC (train==eval on all data): {auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y, pipe.predict(X), digits=3))


if __name__ == "__main__":
    main()
