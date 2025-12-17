"""
Hace PCA 2D sobre embeddings CLIP de imágenes de productos y grafica.

Color:
  - por defecto usa tasa de compra por objeto (page, product_id):
      purchase_rate = mean(bought) sobre todos los sujetos

Entradas:
  - data/processed/image_embeddings/embeddings_index.csv
  - data/processed/products_all_with_images.csv (para calcular purchase_rate)

Salida:
  - data/EDA/eda_results_img/clip_pca.png
  - data/EDA/eda_results_tabular/clip_pca_coords.csv

Uso (desde dataset_neuma):
  python -m dataset_neuma.mains.plot_clip_embeddings_pca --embeddings-dir ./data/processed/image_embeddings --products ./data/processed/products_all_with_images.csv --out-img ./data/EDA/eda_results_img/clip_pca.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Permite ejecución desde carpeta dataset_neuma
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))


def load_embeddings(index_df: pd.DataFrame) -> np.ndarray:
    vecs: List[np.ndarray] = []
    for p in index_df["embedding_path"]:
        vecs.append(np.load(Path(p)))
    return np.stack(vecs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA sobre embeddings CLIP de imágenes de productos.")
    parser.add_argument("--embeddings-dir", type=Path, default=Path("./data/processed/image_embeddings"))
    parser.add_argument("--products", type=Path, default=Path("./data/processed/products_all_with_images.csv"))
    parser.add_argument("--out-img", type=Path, default=Path("./data/EDA/eda_results_img/clip_pca.png"))
    parser.add_argument("--out-csv", type=Path, default=Path("./data/EDA/eda_results_tabular/clip_pca_coords.csv"))
    parser.add_argument("--no-color", action="store_true", help="No colorear por purchase_rate.")
    args = parser.parse_args()

    index_path = args.embeddings_dir / "embeddings_index.csv"
    if not index_path.exists():
        raise SystemExit(f"No se encontró {index_path}")
    index_df = pd.read_csv(index_path)
    if index_df.empty:
        raise SystemExit("embeddings_index.csv está vacío.")

    # Purchase rate por objeto (page, product_id)
    purchase_rate = None
    if not args.no_color:
        if not args.products.exists():
            raise SystemExit(f"No se encontró {args.products}")
        df_prod = pd.read_csv(args.products)
        if not {"page", "product_id", "bought"}.issubset(df_prod.columns):
            raise SystemExit("products_all_with_images.csv debe contener page, product_id, bought")
        purchase_rate = (
            df_prod.dropna(subset=["bought"])
            .groupby(["page", "product_id"])["bought"]
            .mean()
            .rename("purchase_rate")
            .reset_index()
        )
        index_df = index_df.merge(purchase_rate, on=["page", "product_id"], how="left")

    X = load_embeddings(index_df)
    X = np.nan_to_num(X, nan=0.0)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)

    out_coords = index_df.copy()
    out_coords["pc1"] = coords[:, 0]
    out_coords["pc2"] = coords[:, 1]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_coords.to_csv(args.out_csv, index=False)

    plt.figure(figsize=(8, 6))
    if args.no_color:
        plt.scatter(coords[:, 0], coords[:, 1], s=30, alpha=0.8)
    else:
        c = out_coords["purchase_rate"].fillna(0.0).to_numpy()
        sc = plt.scatter(coords[:, 0], coords[:, 1], c=c, cmap="viridis", s=40, alpha=0.85)
        cb = plt.colorbar(sc)
        cb.set_label("purchase_rate")
    plt.title("CLIP image embeddings PCA (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    args.out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_img, dpi=200)
    plt.close()

    print(f"Guardado: {args.out_img}")
    print(f"Guardado: {args.out_csv}")


if __name__ == "__main__":
    main()

