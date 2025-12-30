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

    # derivar llaves compatibles
    def parse_page(s: str):
        return pd.to_numeric(str(s).lower().replace("page", "").strip(), errors="coerce")

    def parse_prod(s: str):
        return pd.to_numeric(str(s).lower().replace("product", "").strip(), errors="coerce")

    base["subject_num"] = pd.to_numeric(base["subject"].astype(str).str.replace("s", "", case=False), errors="coerce")
    base["page_num"] = base["page"].apply(parse_page)
    base["prod_num"] = base["product_id"].apply(parse_prod)

    # manejar faltantes
    base = base.dropna(subset=["subject_num", "page_num", "prod_num"])
    base["subject_num"] = base["subject_num"].astype(int)
    base["page_num"] = base["page_num"].astype(int)
    base["prod_num"] = base["prod_num"].astype(int)
    base["id_prod_key"] = (base["page_num"] - 1) * 24 + base["prod_num"]

    if not {"id_sub", "id_prod"}.issubset(src.columns):
        raise ValueError("El CSV source debe contener columnas id_sub e id_prod.")
    src_small = src[["id_sub", "id_prod"] + [c for c in ["price", "offer", "len_med"] if c in src.columns]].copy()
    src_small = src_small.rename(columns={"id_sub": "subject_num", "id_prod": "id_prod_key"})

    merged = base.merge(src_small, on=["subject_num", "id_prod_key"], how="left")

    # limpiar columnas auxiliares
    merged = merged.drop(columns=["subject_num", "page_num", "prod_num", "id_prod_key"])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"Guardado: {args.out_csv} (filas: {len(merged)})")


if __name__ == "__main__":
    main()
