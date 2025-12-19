"""
Construye un dataset de unión multimodal (imágenes + tabular + EEG concatenado) tomando como referencia products_all_with_images.csv.

Pasos:
  1) Carga products_all_with_images.csv (referencia completa de decisiones).
  2) Agrega embedding_path desde embeddings_index.csv (CLIP).
  3) Hace merge con datos tabulares de sujetos (data_latente_neuma.csv) usando claves:
       subject_norm (S01 -> "1"), id_prod_key = 24*(page_num-1)+prod_num.
  4) Agrega ruta de EEG concatenado por (subject, page, product_id) a partir de eeg_segments_index.csv:
       - Corta segmentos start:end en el npy de cada sujeto.
       - Ordena por start y concatena en el eje temporal.
       - Guarda npy en eeg_concat_dir y anota ruta/shape.
  5) Guarda CSV final con todas las columnas de productos + embedding_path + tabular + eeg_concat_path/eeg_shape.

Uso (desde dataset_neuma):
  python -m mains.build_multimodal_join \
    --products ./data/processed/products_all_with_images.csv \
    --embeddings-dir ./data/processed/image_embeddings \
    --tabular /mnt/otra_particion/home/israel_gpu_data/dataset_neuma/processed/data_latente_neuma.csv \
    --eeg-index ./data/processed/eeg_segments_index.csv \
    --out-csv ./data/processed/multimodal_join.csv \
    --eeg-concat-dir ./data/processed/eeg_concat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Permite ejecución desde carpeta dataset_neuma
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def page_num(val: str) -> int:
    import re

    m = re.match(r"page(\d+)", str(val).lower())
    return int(m.group(1)) if m else None


def prod_num(val: str) -> int:
    import re

    m = re.match(r"product(\d+)", str(val).lower())
    return int(m.group(1)) if m else None


def subj_num(val: str) -> str:
    import re

    m = re.match(r"s0*(\d+)", str(val).lower())
    return m.group(1) if m else str(val)


def concat_eeg_segments(
    df_idx: pd.DataFrame,
    eeg_concat_dir: Path,
    min_duration_s: float = 0.5,
) -> pd.DataFrame:
    """
    df_idx: eeg_segments_index (columnas: subject, page, product_id, npy_path, start, end, bought...)
    Retorna df con columnas subject_norm, page, product_id, eeg_concat_path, eeg_shape
    """
    df = df_idx.copy()
    df.columns = df.columns.str.lower()
    df = df[df["modality"] == "eeg"]
    if "subject" in df.columns:
        df["subject_norm"] = df["subject"].apply(subj_num)
    else:
        raise SystemExit("eeg_segments_index no contiene 'subject'")
    # Para orden, usar start asc
    df = df.sort_values(["subject_norm", "page", "product_id", "start"])

    eeg_concat_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cache: Dict[str, np.ndarray] = {}

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
            # Filtrar por duración mínima (Fs viene en meta?)
            duration_s = None
            if "fs" in df_idx.columns:
                fs_val = df_idx["fs"].dropna().iloc[0]
                duration_s = (seg.shape[1]) / fs_val if fs_val else None
            if duration_s is not None and duration_s < min_duration_s:
                continue
            # Si no hay fs en df_idx, no se puede filtrar; se concatena de todos modos
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
    parser = argparse.ArgumentParser(description="Construye join multimodal (tabular+imagen+EEG concatenado).")
    parser.add_argument("--products", type=Path, default=Path("./data/processed/products_all_with_images.csv"))
    parser.add_argument("--embeddings-dir", type=Path, default=Path("./data/processed/image_embeddings"))
    parser.add_argument("--tabular", type=Path, default=Path("/mnt/otra_particion/home/israel_gpu_data/dataset_neuma/processed/data_latente_neuma.csv"))
    parser.add_argument("--eeg-index", type=Path, default=Path("./data/processed/eeg_segments_index.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("./data/processed/multimodal_join.csv"))
    parser.add_argument("--eeg-concat-dir", type=Path, default=Path("./data/processed/eeg_concat"))
    args = parser.parse_args()

    prod_df = pd.read_csv(args.products)
    prod_df.columns = prod_df.columns.str.lower()
    emb_index = pd.read_csv(args.embeddings_dir / "embeddings_index.csv")
    emb_index.columns = emb_index.columns.str.lower()
    tab_df = pd.read_csv(args.tabular)
    tab_df.columns = tab_df.columns.str.lower()
    if "id_sub" in tab_df.columns:
        tab_df = tab_df.rename(columns={"id_sub": "subject"})

    prod_df["subject_norm"] = prod_df["subject"].astype(str).apply(subj_num)
    tab_df["subject_norm"] = tab_df["subject"].astype(str).apply(lambda s: subj_num(s))
    if "subject" in emb_index.columns:
        emb_index["subject_norm"] = emb_index["subject"].astype(str).apply(subj_num)

    # keys num
    prod_df["page_num"] = prod_df["page"].apply(page_num)
    prod_df["prod_num"] = prod_df["product_id"].apply(prod_num)
    prod_df["id_prod_key"] = prod_df.apply(
        lambda r: 24 * (r["page_num"] - 1) + r["prod_num"] if pd.notna(r["page_num"]) and pd.notna(r["prod_num"]) else np.nan,
        axis=1,
    )

    # merge embeddings (left)
    merged = prod_df.merge(emb_index[["page", "product_id", "embedding_path"]], on=["page", "product_id"], how="left")

    # merge tabular por sujeto + id_prod
    if "id_prod" not in tab_df.columns:
        raise SystemExit("El tabular no contiene 'id_prod' para la llave de producto.")
    merged = merged.merge(
        tab_df,
        left_on=["subject_norm", "id_prod_key"],
        right_on=["subject_norm", "id_prod"],
        how="left",
        suffixes=("", "_tab"),
    )

    # agregar EEG concatenado
    eeg_idx = pd.read_csv(args.eeg_index)
    eeg_concat_df = concat_eeg_segments(eeg_idx, args.eeg_concat_dir)
    merged = merged.merge(
        eeg_concat_df[["subject_norm", "page", "product_id", "eeg_concat_path", "eeg_shape"]],
        on=["subject_norm", "page", "product_id"],
        how="left",
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"Guardado: {args.out_csv} (filas: {len(merged)})")
    print(f"EEG concatenado guardado en: {args.eeg_concat_dir}")


if __name__ == "__main__":
    main()
