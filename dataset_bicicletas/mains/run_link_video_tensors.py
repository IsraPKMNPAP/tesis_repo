from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Ensure package root on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loading.video_windows import map_by_timestamp_or_order


def expand(p: str) -> Path:
    return Path(os.path.expanduser(p))


def main():
    ap = argparse.ArgumentParser(description="Link GPU tensor paths to X_proc_final.pkl and save processed pickle")
    ap.add_argument(
        "--pickle-in",
        type=str,
        default="~/projects/tesis_repo/dataset_bicicletas/data/raw/X_proc_final.pkl",
        help="Ruta al pickle crudo (raw)",
    )
    ap.add_argument(
        "--linux-root",
        type=str,
        default="/mnt/otra_particion/home/israel_gpu_data/video_tensors",
        help="Raíz en GPU/Linux donde viven los window_*.pt",
    )
    ap.add_argument(
        "--timestamp-col",
        type=str,
        default="timestamp",
        help="Columna con timestamp para ordenar/matchear",
    )
    ap.add_argument(
        "--out-pickle",
        type=str,
        default="~/projects/tesis_repo/dataset_bicicletas/data/processed/X_proc_final_linked.pkl",
        help="Salida pickle procesado con columna de rutas GPU",
    )
    ap.add_argument(
        "--out-column",
        type=str,
        default="gpu_tensor_path",
        help="Nombre de la columna con las rutas en GPU",
    )
    args = ap.parse_args()

    inp = expand(args.pickle_in)
    if not inp.exists():
        raise FileNotFoundError(f"No existe el pickle de entrada: {inp}")
    df = pd.read_pickle(inp)

    print("Escaneando y mapeando window_*.pt en:", args.linux_root)
    df_mapped = map_by_timestamp_or_order(df, timestamp_col=args.timestamp_col, gpu_root=args.linux_root)

    # Ensure desired column name
    if args.out_column != "gpu_tensor_path":
        df_mapped = df_mapped.rename(columns={"gpu_tensor_path": args.out_column})

    outp = expand(args.out_pickle)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df_mapped.to_pickle(outp)
    print(f"Guardado: {outp} ({len(df_mapped)} filas) con columna '{args.out_column}'")


if __name__ == "__main__":
    main()

