from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def robust_norm(x: pd.Series, median: float, iqr: float) -> pd.Series:
    if iqr == 0 or np.isnan(iqr):
        return (x - median)
    return (x - median) / iqr


def main() -> None:
    parser = argparse.ArgumentParser(description="Imputa offer/price/len_med en multimodal_join.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/multimodal_join_with_eeg_emb_aug.csv"),
    )
    parser.add_argument(
        "--segments",
        type=Path,
        default=Path("data/processed/eeg_segments_index.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/multimodal_join_with_eeg_emb_aug_imputed.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df.columns = df.columns.str.lower()
    for col in ("subject", "page", "product_id"):
        if col not in df.columns:
            raise ValueError(f"Falta columna {col} en {args.input}")

    # Offer: rellenar con "No"
    if "offer" in df.columns:
        df["offer"] = df["offer"].fillna("No").replace("", "No")
    else:
        df["offer"] = "No"

    # Price: llenar por page+product_id usando mediana
    if "price" in df.columns:
        price_med = (
            df.groupby(["page", "product_id"])["price"]
            .median()
            .reset_index()
            .rename(columns={"price": "price_fill"})
        )
        df = df.merge(price_med, on=["page", "product_id"], how="left")
        df["price"] = df["price"].fillna(df["price_fill"])
        df.drop(columns=["price_fill"], inplace=True)
    else:
        df["price"] = np.nan

    # Len_med: sumar duration_s por subject/page/product_id y normalizar robusto
    seg = pd.read_csv(args.segments)
    seg.columns = seg.columns.str.lower()
    for col in ("subject", "page", "product_id", "duration_s"):
        if col not in seg.columns:
            raise ValueError(f"Falta columna {col} en {args.segments}")
    dur = (
        seg.groupby(["subject", "page", "product_id"])["duration_s"]
        .sum()
        .reset_index()
        .rename(columns={"duration_s": "duration_sum_s"})
    )
    df = df.merge(dur, on=["subject", "page", "product_id"], how="left")

    if "len_med" not in df.columns:
        df["len_med"] = np.nan

    # Estimar mediana e IQR desde duration_sum_s donde len_med existe
    ref = df.loc[df["len_med"].notna() & df["duration_sum_s"].notna(), "duration_sum_s"]
    if ref.empty:
        median_raw = df["duration_sum_s"].median()
        q1 = df["duration_sum_s"].quantile(0.25)
        q3 = df["duration_sum_s"].quantile(0.75)
    else:
        median_raw = ref.median()
        q1 = ref.quantile(0.25)
        q3 = ref.quantile(0.75)
    iqr_raw = q3 - q1

    fill_mask = df["len_med"].isna() & df["duration_sum_s"].notna()
    df.loc[fill_mask, "len_med"] = robust_norm(df.loc[fill_mask, "duration_sum_s"], median_raw, iqr_raw)

    # Reporte e imputacion final
    nan_counts = df[["offer", "price", "len_med"]].isna().sum().to_dict()
    print(f"[impute] NaN counts after fill: {nan_counts}")
    before = len(df)
    df = df.dropna(subset=["offer", "price", "len_med"]).reset_index(drop=True)
    after = len(df)
    print(f"[impute] rows before drop={before} after drop={after}")

    df.to_csv(args.output, index=False)
    print(f"[impute] Saved: {args.output} shape={df.shape}")


if __name__ == "__main__":
    main()
