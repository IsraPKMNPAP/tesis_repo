from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Agrega columnas price/offer/len_med al CSV multimodal_join_with_eeg_emb.")
    parser.add_argument("--base-csv", type=Path, default=Path("./data/processed/multimodal_join_with_eeg_emb.csv"))
    parser.add_argument("--source-csv", type=Path, default=Path("/mnt/otra_particion/home/israel_gpu_data/dataset_neuma/processed/data_latente_estimulo_neuma.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("./data/processed/multimodal_join_with_eeg_emb_aug.csv"))
    args = parser.parse_args()

    base = pd.read_csv(args.base_csv)
    src = pd.read_csv(args.source_csv)

    # normalizar nombres
    base.columns = base.columns.str.lower()
    src.columns = src.columns.str.lower()

    needed = ["subject", "page", "product_id"]
    for col in needed:
        if col not in base.columns or col not in src.columns:
            raise ValueError(f"Falta columna clave '{col}' en base o source.")

    # columnas a traer
    cols_to_add = [c for c in ["price", "offer", "len_med"] if c in src.columns]
    if not cols_to_add:
        raise ValueError("No se encontraron columnas price/offer/len_med en el source CSV.")

    src_small = src[needed + cols_to_add].copy()
    # asegurar claves en str para evitar problemas de tipos
    for col in needed:
        base[col] = base[col].astype(str)
        src_small[col] = src_small[col].astype(str)

    merged = base.merge(src_small, on=needed, how="left")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"Guardado: {args.out_csv} (filas: {len(merged)})")


if __name__ == "__main__":
    main()
