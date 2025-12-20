"""
Construye un dataset de unión multimodal (imágenes + perfiles + EEG concatenado) tomando como referencia products_all_with_images.csv.

Flujo:
  1) products_all_with_images.csv (referencia completa de decisiones).
  2) embeddings_index.csv -> agrega embedding_path.
  3) profiles_all.csv -> merge por subject_norm (S01 -> "1"); elimina columna sucia (eduction/education.1/marital_status duplicada).
  4) eeg_segments_index.csv -> concatena segmentos por (subject, page, product_id), descarta segmentos <0.5s si hay fs.
  5) Marca columnas con <=50 valores únicos como category.
  6) Guarda CSV final con rutas a embeddings y eeg_concat_path/eeg_shape.

Uso (desde dataset_neuma):
  python -m mains.build_multimodal_join \
    --products ./data/processed/products_all_with_images.csv \
    --embeddings-dir ./data/processed/image_embeddings \
    --profiles ./data/processed/profiles_all.csv \
    --eeg-index ./data/processed/eeg_segments_index.csv \
    --out-csv ./data/processed/multimodal_join.csv \
    --eeg-concat-dir ./data/processed/eeg_concat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import re

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def page_num(val: str) -> int:
    m = re.match(r"page(\d+)", str(val).lower())
    return int(m.group(1)) if m else None


def prod_num(val: str) -> int:
    m = re.match(r"product(\d+)", str(val).lower())
    return int(m.group(1)) if m else None


def subj_num(val: str) -> str:
    m = re.match(r"s0*(\d+)", str(val).lower())
    return m.group(1) if m else str(val)


def concat_eeg_segments(df_idx: pd.DataFrame, eeg_concat_dir: Path, min_duration_s: float = 0.5) -> pd.DataFrame:
    df = df_idx.copy()
    df.columns = df.columns.str.lower()
    if "subject" not in df.columns:
        raise SystemExit("eeg_segments_index no contiene 'subject'")
    df["subject_norm"] = df["subject"].apply(subj_num)
    df = df.sort_values(["subject_norm", "page", "product_id", "start"])

    eeg_concat_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cache: Dict[str, np.ndarray] = {}

    fs_global = None
    if "fs" in df.columns and df["fs"].notna().any():
        fs_global = df["fs"].dropna().iloc[0]

    for (subj, page, prod), grp in df.groupby(["subject_norm", "page", "product_id"]):
        segments = []
        for _, r in grp.iterrows():
            npy_path = r["npy_path"]
            if npy_path not in cache:
                cache[npy_path] = np.load(npy_path)
            arr = cache[npy_path]
            start, end = int(r["start"]), int(r["end"])
            end = min(end, arr.shape[1] - 1)
            seg = arr[:, start : end + 1]
            if fs_global:
                duration_s = seg.shape[1] / fs_global
                if duration_s < min_duration_s:
                    continue
            segments.append(seg)
        if not segments:
            continue
        concat = np.concatenate(segments, axis=1)
        out_path = eeg_concat_dir / f"{subj}_{page}_{prod}.npy"
        np.save(out_path, concat.astype(np.float32))
        rows.append(
            {
                "subject_norm": subj,
                "page": page,
                "product_id": prod,
                "eeg_concat_path": str(out_path),
                "eeg_shape": str(list(concat.shape)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye join multimodal (imagen + perfiles + EEG concatenado).")
    parser.add_argument("--products", type=Path, default=Path("./data/processed/products_all_with_images.csv"))
    parser.add_argument("--embeddings-dir", type=Path, default=Path("./data/processed/image_embeddings"))
    parser.add_argument("--profiles", type=Path, default=Path("./data/processed/profiles_all.csv"))
    parser.add_argument("--eeg-index", type=Path, default=Path("./data/processed/eeg_segments_index.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("./data/processed/multimodal_join.csv"))
    parser.add_argument("--eeg-concat-dir", type=Path, default=Path("./data/processed/eeg_concat"))
    args = parser.parse_args()

    # Carga y diagnóstico inicial
    prod_df = pd.read_csv(args.products)
    prod_df.columns = prod_df.columns.str.lower()
    emb_index = pd.read_csv(args.embeddings_dir / "embeddings_index.csv")
    emb_index.columns = emb_index.columns.str.lower()
    prof_df = pd.read_csv(args.profiles)
    prof_df.columns = prof_df.columns.str.lower()

    print(f"[diag] products rows={len(prod_df)} cols={len(prod_df.columns)}")
    print(f"[diag] embeddings_index rows={len(emb_index)} cols={len(emb_index.columns)}")
    print(f"[diag] profiles rows={len(prof_df)} cols={len(prof_df.columns)}")

    # Limpiar columnas sucias en perfiles
    for bad in ["eduction", "education.1", "marital_status"]:
        if bad in prof_df.columns:
            prof_df = prof_df.drop(columns=[bad])

    # Normalización de llaves
    prod_df["subject_norm"] = prod_df["subject"].astype(str).apply(subj_num)
    prof_df["subject_norm"] = prof_df["subject"].astype(str).apply(subj_num)
    if "subject" in emb_index.columns:
        emb_index["subject_norm"] = emb_index["subject"].astype(str).apply(subj_num)

    prod_df["page_num"] = prod_df["page"].apply(page_num)
    prod_df["prod_num"] = prod_df["product_id"].apply(prod_num)

    # Merge embeddings
    merged = prod_df.merge(emb_index[["page", "product_id", "embedding_path"]], on=["page", "product_id"], how="left")

    # Merge perfiles por subject_norm
    merged = merged.merge(prof_df, on="subject_norm", how="left", suffixes=("", "_prof"))

    # Agregar EEG concatenado
    eeg_idx = pd.read_csv(args.eeg_index)
    eeg_concat_df = concat_eeg_segments(eeg_idx, args.eeg_concat_dir)
    merged = merged.merge(
        eeg_concat_df[["subject_norm", "page", "product_id", "eeg_concat_path", "eeg_shape"]],
        on=["subject_norm", "page", "product_id"],
        how="left",
    )

    # Convertir columnas con <=50 valores únicos a category
    for col in merged.columns:
        if col in ["subject_norm", "subject", "page", "product_id", "embedding_path", "eeg_concat_path", "eeg_shape"]:
            continue
        try:
            if merged[col].nunique(dropna=True) <= 50:
                merged[col] = merged[col].astype("category")
        except Exception:
            pass

    # Eliminar columnas duplicadas residuales
    drop_cols = [c for c in merged.columns if c.endswith(".1") or c.endswith("_prof")]
    for dc in drop_cols:
        if dc in merged.columns and dc.replace(".1", "") in merged.columns:
            merged = merged.drop(columns=[dc])

    # Diagnóstico de nulos clave
    print(f"[diag] rows after merge: {len(merged)}")
    for key in ["embedding_path", "eeg_concat_path", "education", "maritalstatus"]:
        if key in merged.columns:
            print(f"[diag] nulls {key}: {merged[key].isna().sum()}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"Guardado: {args.out_csv} (filas: {len(merged)})")
    print(f"EEG concatenado guardado en: {args.eeg_concat_dir}")


if __name__ == "__main__":
    main()

