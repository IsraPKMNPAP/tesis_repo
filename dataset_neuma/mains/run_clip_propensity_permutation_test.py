"""
Permutation test para evaluar si los embeddings CLIP contienen información sobre propensión de compra.

Se replica el setup de `run_clip_propensity_logreg.py`:
  - purchase_rate = mean(bought) por objeto (page, product_id)
  - y = 1 si purchase_rate >= Q3 (p75)
  - y = 0 si purchase_rate <= Q1 (p25)
  - ignora el resto

Luego:
  - AUC observado: entrenando y evaluando en todos los datos (train==eval), como en tu criterio exploratorio.
  - Permutation test: permuta y, recalcula AUC n_perm veces.
  - p-valor empírico: (#{AUC_perm >= AUC_obs}+1)/(n_perm+1)
  - Guarda histograma y opcionalmente el vector de AUC permutados.

Uso (desde dataset_neuma):
  python -m dataset_neuma.mains.run_clip_propensity_permutation_test --embeddings-dir ./data/processed/image_embeddings --products ./data/processed/products_all_with_images.csv --n-perm 1000 --C 0.1 --seed 42 --out-img ./data/EDA/eda_results_img/clip_auc_permtest.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Permite ejecución desde carpeta dataset_neuma
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))


def load_embeddings(paths: List[str]) -> np.ndarray:
    vecs = [np.load(Path(p)).astype(np.float32) for p in paths]
    return np.stack(vecs, axis=0)


def compute_auc_cv(X: np.ndarray, y: np.ndarray, C: float, seed: int, n_splits: int) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        clf = LogisticRegression(
            penalty="l2",
            C=C,
            solver="liblinear",
            class_weight="balanced",
            max_iter=500,
        )
        clf.fit(X[train_idx], y[train_idx])
        y_score = clf.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], y_score))
    return float(np.mean(aucs))


def permutation_test_auc_cv(X: np.ndarray, y: np.ndarray, n_perm: int, C: float, seed: int, n_splits: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    auc_perm = np.zeros(n_perm, dtype=float)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        auc_perm[i] = compute_auc_cv(X, y_perm, C=C, seed=seed, n_splits=n_splits)
        if (i + 1) % 50 == 0:
            print(f"  perm {i+1}/{n_perm} ...")
    return auc_perm


def main() -> None:
    parser = argparse.ArgumentParser(description="Permutation test de AUC para embeddings CLIP.")
    parser.add_argument("--embeddings-dir", type=Path, default=Path("./data/processed/image_embeddings"))
    parser.add_argument("--products", type=Path, default=Path("./data/processed/products_all_with_images.csv"))
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--C", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--out-img", type=Path, default=Path("./data/EDA/eda_results_img/clip_auc_permtest.png"))
    parser.add_argument("--out-npy", type=Path, default=Path("./data/EDA/eda_results_tabular/clip_auc_permtest.npy"))
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

    df = emb_index.merge(rates, on=["page", "product_id"], how="left").dropna(subset=["purchase_rate"])
    q1 = float(df["purchase_rate"].quantile(0.25))
    q3 = float(df["purchase_rate"].quantile(0.75))
    df["y"] = np.where(df["purchase_rate"] >= q3, 1, np.where(df["purchase_rate"] <= q1, 0, np.nan))
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)

    if df["y"].nunique() < 2:
        raise SystemExit("No hay suficientes clases luego del filtrado por cuartiles.")

    X = load_embeddings(df["embedding_path"].tolist())
    y = df["y"].to_numpy()

    auc_obs = compute_auc_cv(X, y, C=args.C, seed=args.seed, n_splits=args.n_splits)
    print("== Permutation test (AUC, Stratified CV) ==")
    print(f"Q1={q1:.4f}  Q3={q3:.4f}")
    print(f"Used samples: {len(df)} (class0={int((y==0).sum())}, class1={int((y==1).sum())})")
    print(f"AUC observed (mean CV={args.n_splits}): {auc_obs:.4f}")

    auc_perm = permutation_test_auc_cv(X, y, n_perm=args.n_perm, C=args.C, seed=args.seed, n_splits=args.n_splits)
    p_emp = (float(np.sum(auc_perm >= auc_obs)) + 1.0) / (len(auc_perm) + 1.0)
    print(f"Empirical p-value: {p_emp:.6f}")

    args.out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(auc_perm, bins=30, alpha=0.75, color="gray", edgecolor="black")
    plt.axvline(auc_obs, color="red", linestyle="--", linewidth=2, label=f"AUC observed = {auc_obs:.3f}")
    plt.xlabel("AUC under permutation")
    plt.ylabel("Frequency")
    plt.title(f"Permutation test (n={args.n_perm})\\nempirical p={p_emp:.6f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_img, dpi=200)
    plt.close()
    print(f"Guardado histograma: {args.out_img}")

    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, auc_perm.astype(np.float32))
    print(f"Guardado auc_perm: {args.out_npy}")


if __name__ == "__main__":
    main()
