"""
Runner para exportar todos los participantes NEUMA.

Lee .mat desde un directorio RAW (en GPU) y escribe:
  - Pesados (EEG/ET) en PROCESSED externo: /mnt/otra_particion/home/israel_gpu_data/dataset_neuma/processed
  - CSV ligeros en el repo: data/processed (en este proyecto)

Uso (en GPU):
    python -m dataset_neuma.mains.run_export_all \\
        --raw-dir /mnt/otra_particion/home/israel_gpu_data/dataset_neuma/raw \\
        --processed-dir /mnt/otra_particion/home/israel_gpu_data/dataset_neuma/processed \\
        --repo-processed-dir /path/al/repo/dataset_neuma/data/processed \\
        --screen-width 1920 --screen-height 1080
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecución desde la carpeta dataset_neuma (añade el padre al sys.path)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))

from dataset_neuma.utils.exporters import find_participants, process_participant


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta todos los sujetos NEUMA a formatos ligeros/pesados.")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directorio con .mat crudos (en GPU).")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
        help="Directorio donde guardar EEG/ET pesados (fuera del repo, ej. /mnt/otra_particion/.../processed).",
    )
    parser.add_argument(
        "--repo-processed-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "processed",
        help="Directorio en el repo para CSV/JSON ligeros (segments, products, profile).",
    )
    parser.add_argument("--screen-width", type=int, default=1920, help="Ancho de pantalla (px) para ET.")
    parser.add_argument("--screen-height", type=int, default=1080, help="Alto de pantalla (px) para ET.")
    parser.add_argument("--pattern", type=str, default="S*.mat", help="Patrón glob para sujetos (ej. S*.mat).")
    args = parser.parse_args()

    mat_files = find_participants(args.raw_dir, args.pattern)
    if not mat_files:
        raise SystemExit(f"No se encontraron .mat en {args.raw_dir} con patrón {args.pattern}")

    print(f"Encontrados {len(mat_files)} participantes.")
    for mat_path in mat_files:
        subject = mat_path.stem
        print(f"Procesando {subject} ...")
        process_participant(
            mat_path=mat_path,
            subject=subject,
            out_processed_root=args.processed_dir,
            out_repo_processed_root=args.repo_processed_dir,
            screen_w=args.screen_width,
            screen_h=args.screen_height,
        )
    print("Listo.")


if __name__ == "__main__":
    main()
