"""
ANOVA univariado para cada feature EEG vs etiqueta de compra.

Entrada:
  - data/EDA/eda_results_tabular/eeg_band_features_wide.csv

Salidas (data/EDA/eda_results_tabular):
  - anova_results.csv (feature, F, p, p_fdr, eta2, mean_b0, mean_b1, channel, band, metric)
  - anova_channel_summary.csv (resumen por canal)
  - anova_channel_band_summary.csv (resumen por canal+banda)

Uso (desde dataset_neuma):
  python -m dataset_neuma.mains.run_anova \
    --input ./data/EDA/eda_results_tabular/eeg_band_features_wide.csv \
    --out-dir ./data/EDA/eda_results_tabular
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests

# Permite ejecución desde carpeta dataset_neuma
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))


PATTERN = re.compile(r"^(ch\d+)_(theta|alpha|beta|gamma_low)_(mean|std|rel)$")


def split_feature_name(name: str) -> Tuple[str, str, str]:
    m = PATTERN.match(name)
    if not m:
        return "", "", ""
    return m.group(1), m.group(2), m.group(3)


def anova_univariate(df: pd.DataFrame, label_col: str, feature_cols: List[str]) -> pd.DataFrame:
    y = df[label_col].values
    g0 = df.loc[y == 0]
    g1 = df.loc[y == 1]

    rows = []
    for feat in feature_cols:
        x0 = g0[feat].values
        x1 = g1[feat].values
        F, p = f_oneway(x0, x1)

        overall_mean = df[feat].mean()
        ss_between = len(x0) * (x0.mean() - overall_mean) ** 2 + len(x1) * (x1.mean() - overall_mean) ** 2
        ss_total = ((df[feat] - overall_mean) ** 2).sum()
        eta2 = float(ss_between / ss_total) if ss_total > 0 else 0.0

        rows.append((feat, F, p, eta2, x0.mean(), x1.mean()))

    out = pd.DataFrame(rows, columns=["feature", "F", "p", "eta2", "mean_b0", "mean_b1"])
    out["p_fdr"] = multipletests(out["p"].values, method="fdr_bh")[1]
    out[["channel", "band", "metric"]] = out["feature"].apply(lambda s: pd.Series(split_feature_name(s)))
    return out.sort_values("eta2", ascending=False)


def summarize_channel(anova_df: pd.DataFrame) -> pd.DataFrame:
    return (
        anova_df.groupby("channel")
        .agg(
            eta2_max=("eta2", "max"),
            eta2_mean=("eta2", "mean"),
            n_sig_fdr=("p_fdr", lambda x: int((x < 0.05).sum())),
            n_features=("feature", "count"),
        )
        .sort_values(["eta2_max", "eta2_mean"], ascending=False)
    )


def summarize_channel_band(anova_df: pd.DataFrame) -> pd.DataFrame:
    return (
        anova_df.groupby(["channel", "band"])
        .agg(
            eta2_max=("eta2", "max"),
            eta2_mean=("eta2", "mean"),
            n_sig_fdr=("p_fdr", lambda x: int((x < 0.05).sum())),
            n_features=("feature", "count"),
        )
        .sort_values(["eta2_max", "eta2_mean"], ascending=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ANOVA univariado de features EEG vs etiqueta de compra.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./data/EDA/eda_results_tabular/eeg_band_features_wide.csv"),
        help="CSV de entrada (formato ancho).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./data/EDA/eda_results_tabular"),
        help="Directorio de salida para resultados ANOVA.",
    )
    parser.add_argument("--label-col", type=str, default="bought", help="Nombre de la columna de etiqueta.")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"No se encontró {args.input}")

    df_wide = pd.read_csv(args.input)
    if args.label_col not in df_wide.columns:
        raise SystemExit(f"No se encuentra la columna de etiqueta {args.label_col} en {args.input}")

    # Selección de columnas EEG
    eeg_cols = [c for c in df_wide.columns if PATTERN.match(c)]
    if not eeg_cols:
        raise SystemExit("No se encontraron columnas EEG que cumplan el patrón esperado.")

    df = df_wide[[args.label_col] + eeg_cols].copy()
    df = df.dropna(subset=[args.label_col])
    df[eeg_cols] = df[eeg_cols].fillna(df[eeg_cols].median(numeric_only=True))
    df[args.label_col] = df[args.label_col].astype(int)

    anova_df = anova_univariate(df, args.label_col, eeg_cols)
    chan_summary = summarize_channel(anova_df)
    chan_band_summary = summarize_channel_band(anova_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    anova_path = args.out_dir / "anova_results.csv"
    chan_path = args.out_dir / "anova_channel_summary.csv"
    chan_band_path = args.out_dir / "anova_channel_band_summary.csv"

    anova_df.to_csv(anova_path, index=False)
    chan_summary.to_csv(chan_path)
    chan_band_summary.to_csv(chan_band_path)

    print(f"Guardado: {anova_path} (features: {len(anova_df)})")
    print(f"Guardado: {chan_path}")
    print(f"Guardado: {chan_band_path}")
    print("\nTop 5 features por eta2:")
    print(anova_df.head(5)[["feature", "eta2", "p", "p_fdr", "mean_b0", "mean_b1"]])


if __name__ == "__main__":
    main()
