"""
Construye un índice único de segmentos EEG con etiqueta de compra.

Genera un CSV en data/processed con columnas:
  subject, page, product_id, bought, npy_path, start, end, start_time_s, end_time_s, duration_s, fs, shape

Asume:
  - products_all.csv en data/processed (agregado).
  - SXX_segments_with_times.csv en data/processed por sujeto.
  - Archivos EEG npy en /mnt/otra_particion/home/israel_gpu_data/dataset_neuma/processed/eeg/{SXX}_eeg_data.npy
  - Metas EEG en .../processed/eeg/{SXX}_eeg_meta.json (opcional para fs/shape).

Uso (desde dataset_neuma):
  python -m dataset_neuma.mains.build_eeg_segments_index \
    --processed-dir ./data/processed \
    --eeg-dir /mnt/otra_particion/home/israel_gpu_data/dataset_neuma/processed/eeg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Permite ejecución desde carpeta dataset_neuma
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye índice de segmentos EEG con etiqueta de compra.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "processed",
        help="Directorio del repo con products_all.csv y segmentos por sujeto.",
    )
    parser.add_argument(
        "--eeg-dir",
        type=Path,
        default=Path("/mnt/otra_particion/home/israel_gpu_data/dataset_neuma/processed/eeg"),
        help="Directorio externo con los npy/meta de EEG.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Archivo CSV de salida (por defecto processed-dir/eeg_segments_index.csv)",
    )
    args = parser.parse_args()

    processed_dir = args.processed_dir
    output_path = args.output or (processed_dir / "eeg_segments_index.csv")

    products_all = processed_dir / "products_all.csv"
    if not products_all.exists():
        raise SystemExit(f"No se encontró products_all.csv en {processed_dir}")
    df_products = pd.read_csv(products_all)

    # Esperamos columnas: subject, page, product_id, bought, ...
    required_cols = {"subject", "page", "product_id", "bought"}
    if not required_cols.issubset(df_products.columns):
        raise SystemExit(f"products_all.csv no contiene columnas requeridas {required_cols}")

    subjects = sorted(df_products["subject"].unique())
    rows = []
    missing_segments = []

    for subj in subjects:
        seg_path = processed_dir / f"{subj}_segments_with_times.csv"
        if not seg_path.exists():
            missing_segments.append(subj)
            continue
        df_seg = pd.read_csv(seg_path)
        df_seg = df_seg[df_seg["modality"] == "EEG"]

        # Merge con productos para traer bought
        df_subj_prod = df_products[df_products["subject"] == subj]
        df_merge = df_seg.merge(
            df_subj_prod[["subject", "page", "product_id", "bought"]],
            on=["subject", "page", "product_id"],
            how="left",
            validate="m:m",
        )

        npy_path = args.eeg_dir / f"{subj}_eeg_data.npy"
        meta_path = args.eeg_dir / f"{subj}_eeg_meta.json"
        meta = load_json(meta_path)
        fs = meta.get("Fs")
        shape = meta.get("shape")

        for _, r in df_merge.iterrows():
            rows.append(
                {
                    "subject": subj,
                    "page": r["page"],
                    "product_id": r["product_id"],
                    "bought": r.get("bought"),
                    "npy_path": str(npy_path),
                    "start": r["start"],
                    "end": r["end"],
                    "start_time_s": r.get("start_time_s"),
                    "end_time_s": r.get("end_time_s"),
                    "duration_s": r.get("duration_s"),
                    "fs": fs,
                    "shape": shape,
                }
            )

    df_out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)

    print(f"Guardado: {output_path} (filas: {len(df_out)})")
    if missing_segments:
        print(f"Sin segments_with_times.csv para: {missing_segments}")


if __name__ == "__main__":
    main()
