"""
Hace PCA 2D sobre embeddings CLIP de imágenes de productos y grafica.

Color:
  - por defecto usa tasa de compra por objeto (page, product_id):
      purchase_rate = mean(bought) sobre todos los sujetos

Anotaciones:
  - opcionalmente anota un producto de referencia y sus vecinos más cercanos en PCA,
    usando el nombre (columna `description`) desde products_all_with_images.csv.
  - Para evitar superposición, los nombres se listan en una columna fuera del eje
    y se conectan con líneas punteadas y semi-transparentes hacia cada punto.

Entradas:
  - data/processed/image_embeddings/embeddings_index.csv
  - data/processed/products_all_with_images.csv (para purchase_rate y description)

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
from typing import List

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
    parser.add_argument("--no-labels", action="store_true", help="No anotar productos (aunque se entregue ref).")
    parser.add_argument("--ref-page", type=str, default=None, help="Página de referencia (ej: Page1).")
    parser.add_argument("--ref-product", type=str, default=None, help="Producto de referencia (ej: Product1).")
    parser.add_argument("--radius", type=float, default=0.5, help="Radio en el espacio PCA para anotar vecinos.")
    parser.add_argument("--top-k", type=int, default=10, help="Si no hay vecinos dentro del radio, anota K más cercanos.")
    args = parser.parse_args()

    index_path = args.embeddings_dir / "embeddings_index.csv"
    if not index_path.exists():
        raise SystemExit(f"No se encontró {index_path}")
    index_df = pd.read_csv(index_path)
    if index_df.empty:
        raise SystemExit("embeddings_index.csv está vacío.")

    need_products = (not args.no_color) or ((not args.no_labels) and args.ref_page and args.ref_product)
    if need_products:
        if not args.products.exists():
            raise SystemExit(f"No se encontró {args.products}")
        df_prod = pd.read_csv(args.products)
        if not {"page", "product_id", "bought"}.issubset(df_prod.columns):
            raise SystemExit("products_all_with_images.csv debe contener page, product_id, bought")

        if not args.no_color:
            purchase_rate = (
                df_prod.dropna(subset=["bought"])
                .groupby(["page", "product_id"])["bought"]
                .mean()
                .rename("purchase_rate")
                .reset_index()
            )
            index_df = index_df.merge(purchase_rate, on=["page", "product_id"], how="left")

        if "description" in df_prod.columns:
            desc_df = (
                df_prod.dropna(subset=["description"])
                .groupby(["page", "product_id"])["description"]
                .first()
                .rename("description")
                .reset_index()
            )
            index_df = index_df.merge(desc_df, on=["page", "product_id"], how="left")

    X = load_embeddings(index_df)
    X = np.nan_to_num(X, nan=0.0)
    Xs = StandardScaler().fit_transform(X)
    coords = PCA(n_components=2, random_state=42).fit_transform(Xs)

    out_coords = index_df.copy()
    out_coords["pc1"] = coords[:, 0]
    out_coords["pc2"] = coords[:, 1]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_coords.to_csv(args.out_csv, index=False)

    plt.figure(figsize=(9, 6))
    ax = plt.gca()

    if args.no_color:
        ax.scatter(coords[:, 0], coords[:, 1], s=30, alpha=0.8)
    else:
        c = out_coords.get("purchase_rate", pd.Series([0.0] * len(out_coords))).fillna(0.0).to_numpy()
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=c, cmap="viridis", s=40, alpha=0.85)
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("purchase_rate")

    # Anotar vecinos cercanos para una referencia
    if (not args.no_labels) and args.ref_page and args.ref_product:
        mask_ref = (out_coords["page"] == args.ref_page) & (out_coords["product_id"] == args.ref_product)
        if mask_ref.any():
            ref_idx = int(np.where(mask_ref.to_numpy())[0][0])
            ref_pt = coords[ref_idx]
            dists = np.sqrt(np.sum((coords - ref_pt) ** 2, axis=1))
            within = np.where(dists <= args.radius)[0].tolist()
            if ref_idx not in within:
                within.append(ref_idx)
            if len(within) <= 1:
                within = np.argsort(dists)[: max(2, args.top_k)].tolist()
            within = sorted(within, key=lambda i: float(dists[i]))[: max(2, args.top_k)]

            ax.scatter([ref_pt[0]], [ref_pt[1]], s=140, facecolors="none", edgecolors="black", linewidths=2)

            # Etiquetas en columna fuera del eje (derecha)
            ys = np.linspace(0.92, 0.08, len(within))
            x_text = 1.02

            for slot, i in enumerate(within):
                desc = out_coords.loc[i, "description"] if "description" in out_coords.columns else None
                label = str(desc) if pd.notna(desc) else "Unknown"
                is_ref = i == ref_idx
                ax.annotate(
                    label,
                    xy=(float(coords[i, 0]), float(coords[i, 1])),
                    xycoords="data",
                    xytext=(x_text, float(ys[slot])),
                    textcoords="axes fraction",
                    ha="left",
                    va="center",
                    fontsize=9 if is_ref else 8,
                    fontweight="bold" if is_ref else "normal",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=1.0),
                    arrowprops=dict(arrowstyle="-", linestyle="--", color="black", lw=1.0, alpha=0.30),
                    clip_on=False,
                )
        else:
            print(f"[WARN] Referencia no encontrada: {args.ref_page} {args.ref_product}")

    ax.set_title("CLIP image embeddings PCA (2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    args.out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_img, dpi=200)
    plt.close()

    print(f"Guardado: {args.out_img}")
    print(f"Guardado: {args.out_csv}")


if __name__ == "__main__":
    main()

