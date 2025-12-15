"""
Genera proyecciones PCA y UMAP del dataset ancho de EEG y guarda las imágenes.

Entrada:
  - data/EDA/eda_results_tabular/eeg_band_features_wide.csv
Salida:
  - data/EDA/eda_results_img/pca.png
  - data/EDA/eda_results_img/umap.png

Uso (desde dataset_neuma):
  python -m dataset_neuma.mains.run_pca_umap \
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

# Permite ejecución desde carpeta dataset_neuma
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    id_cols = ["subject", "page", "product_id", "bought"]
    feat_cols = [c for c in df.columns if c not in id_cols]
    X = df[feat_cols].copy()
    X = X.fillna(0.0)
    X = X.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    meta = df[id_cols].reset_index(drop=True)
    return meta, X_scaled, X.columns


def plot_embedding(embedding: np.ndarray, meta: pd.DataFrame, title: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    colors = meta["bought"].fillna(-1).astype(int)
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap="viridis", s=10, alpha=0.7)
    # Discretizar la barra: ticks en valores únicos de clase
    unique_vals = sorted(colors.unique())
    cbar = plt.colorbar(scatter, ticks=unique_vals)
    cbar.ax.set_yticklabels([str(v) for v in unique_vals])
    cbar.set_label("bought (NaN=-1)")
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA y UMAP sobre EEG ancho.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./data/EDA/eda_results_tabular/eeg_band_features_wide.csv"),
        help="CSV de entrada en formato ancho.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./data/EDA/eda_results_img"),
        help="Directorio donde guardar las imágenes.",
    )
    parser.add_argument("--pca-components", type=int, default=2, help="Componentes PCA.")
    parser.add_argument("--umap-components", type=int, default=2, help="Componentes UMAP.")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"No se encontró {args.input}")

    df = pd.read_csv(args.input)
    meta, X_scaled, _ = prepare_data(df)

    # PCA
    pca = PCA(n_components=args.pca_components, random_state=42)
    emb_pca = pca.fit_transform(X_scaled)
    plot_embedding(emb_pca, meta, "PCA (colored by bought)", args.out_dir / "pca.png")

    # UMAP
    reducer = umap.UMAP(n_components=args.umap_components, random_state=42, n_jobs=1)
    emb_umap = reducer.fit_transform(X_scaled)
    plot_embedding(emb_umap, meta, "UMAP (colored by bought)", args.out_dir / "umap.png")

    print(f"Guardado: {args.out_dir / 'pca.png'}")
    print(f"Guardado: {args.out_dir / 'umap.png'}")


if __name__ == "__main__":
    main()
