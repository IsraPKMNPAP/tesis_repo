"""
Pivotea eeg_band_features.csv a un formato ancho para EDA/PCA.

Entrada:
  - data/EDA/eda_results_tabular/eeg_band_features.csv
    (columnas: subject, page, product_id, bought, channel_idx, band, power_mean, power_std, power_rel)

Salida:
  - data/EDA/eda_results_tabular/eeg_band_features_wide.csv
    Índice: subject, page, product_id, bought
    Columnas: {channel_idx}_{band}_{stat} para stat en [mean, std, rel]

Uso (desde dataset_neuma):
  python -m dataset_neuma.mains.pivot_eeg_band_features \
    --input ./data/EDA/eda_results_tabular/eeg_band_features.csv \
    --output ./data/EDA/eda_results_tabular/eeg_band_features_wide.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Permite ejecución desde carpeta dataset_neuma
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pivotea features EEG por banda/canal a formato ancho.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./data/EDA/eda_results_tabular/eeg_band_features.csv"),
        help="CSV de entrada en formato largo.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/EDA/eda_results_tabular/eeg_band_features_wide.csv"),
        help="CSV de salida en formato ancho.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"No se encontró {args.input}")

    df = pd.read_csv(args.input)
    required = {"subject", "page", "product_id", "bought", "channel_idx", "band", "power_mean", "power_std", "power_rel"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Faltan columnas requeridas: {required - set(df.columns)}")

    df["channel_idx"] = df["channel_idx"].astype(int)
    # Crear columnas compuestas
    df_long = df.melt(
        id_vars=["subject", "page", "product_id", "bought", "channel_idx", "band"],
        value_vars=["power_mean", "power_std", "power_rel"],
        var_name="stat",
        value_name="value",
    )
    df_long["col"] = df_long.apply(lambda r: f"ch{r.channel_idx}_{r.band}_{r.stat.split('_')[1] if '_' in r.stat else r.stat}", axis=1)

    df_pivot = df_long.pivot_table(
        index=["subject", "page", "product_id", "bought"],
        columns="col",
        values="value",
        aggfunc="mean",
    ).reset_index()

    # Ordenar columnas: id primero, luego features ordenados
    id_cols = ["subject", "page", "product_id", "bought"]
    feat_cols = sorted([c for c in df_pivot.columns if c not in id_cols])
    df_pivot = df_pivot[id_cols + feat_cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_pivot.to_csv(args.output, index=False)
    print(f"Guardado: {args.output} (filas: {len(df_pivot)}, cols: {len(df_pivot.columns)})")


if __name__ == "__main__":
    main()
