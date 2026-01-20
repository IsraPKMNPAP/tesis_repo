from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_page(val: str) -> int:
    if pd.isna(val):
        return -1
    s = str(val).strip().lower()
    if s.startswith("page"):
        s = s.replace("page", "")
    return int(float(s))


def parse_product(val: str) -> int:
    if pd.isna(val):
        return -1
    s = str(val).strip().lower()
    if s.startswith("product"):
        s = s.replace("product", "")
    return int(float(s))


def subject_to_num(val: str) -> int:
    if pd.isna(val):
        return -1
    s = str(val).strip().upper()
    if s.startswith("S"):
        s = s[1:]
    return int(float(s))


def main() -> None:
    parser = argparse.ArgumentParser(description="Join latente (left) con dataset multimodal.")
    parser.add_argument(
        "--latente",
        type=Path,
        default=Path("/mnt/otra_particion/home/israel_gpu_data/dataset_neuma/processed/data_latente_estimulo_neuma_age_robust.csv"),
    )
    parser.add_argument(
        "--multimodal",
        type=Path,
        default=Path("data/processed/multimodal_join_with_eeg_emb_aug_imputed.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/multimodal_join_with_eeg_emb_aug_imputed_latente_left.csv"),
    )
    args = parser.parse_args()

    lat = pd.read_csv(args.latente)
    mm = pd.read_csv(args.multimodal)
    lat.columns = lat.columns.str.lower()
    mm.columns = mm.columns.str.lower()

    if "id_sub" not in lat.columns or "id_prod" not in lat.columns:
        raise ValueError("Latente debe tener columnas id_sub e id_prod.")
    if not {"subject", "page", "product_id"}.issubset(mm.columns):
        raise ValueError("Multimodal debe tener columnas subject, page, product_id.")

    lat["id_sub_num"] = pd.to_numeric(lat["id_sub"], errors="coerce").astype("Int64")
    lat["id_prod_num"] = pd.to_numeric(lat["id_prod"], errors="coerce").astype("Int64")

    for col in ["subject_num", "page_num", "prod_num", "id_prod_num"]:
        if col in mm.columns:
            mm = mm.drop(columns=[col])
    mm["subject_num"] = mm["subject"].apply(subject_to_num).astype("Int64")
    mm["page_num"] = mm["page"].apply(parse_page).astype("Int64")
    mm["prod_num"] = mm["product_id"].apply(parse_product).astype("Int64")
    mm["id_prod_num"] = ((mm["page_num"] - 1) * 24 + mm["prod_num"]).astype("Int64")

    join_cols = [c for c in mm.columns if c not in {"subject", "page", "product_id", "subject_num", "id_prod_num"}]
    merged = lat.merge(
        mm[join_cols + ["subject_num", "id_prod_num"]],
        how="left",
        left_on=["id_sub_num", "id_prod_num"],
        right_on=["subject_num", "id_prod_num"],
        suffixes=("", "_mm"),
    )

    same_cols = set(lat.columns) & set(mm.columns)
    print(f"[join] shared columns count: {len(same_cols)}")
    print(f"[join] output rows: {len(merged)} cols: {len(merged.columns)}")
    if merged.isna().any().any():
        na_counts = merged.isna().sum()
        top_na = na_counts[na_counts > 0].sort_values(ascending=False).head(20)
        print("[join] NaN columns (top 20):")
        print(top_na.to_string())

    # Drop rows with any NaN after join (left is latente)
    before = len(merged)
    merged = merged.dropna().reset_index(drop=True)
    if len(merged) != before:
        print(f"[join] dropped rows with NaN: {before - len(merged)}")

    # Normalize column names by removing suffixes
    rename_map = {}
    for c in merged.columns:
        if c.endswith("_mm"):
            base = c[:-3]
            if base not in merged.columns:
                rename_map[c] = base
    if rename_map:
        merged = merged.rename(columns=rename_map)
        print(f"[join] renamed {len(rename_map)} columns (removed _mm)")

    print(f"[join] final rows: {len(merged)} cols: {len(merged.columns)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
