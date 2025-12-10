"""
Agrega CSV ligeros de todos los participantes ya exportados.

Toma todos los archivos *_profile_demographics.csv y *_products.csv
desde data/processed (o un directorio dado) y genera:
  - profiles_all.csv
  - products_all.csv

Uso:
    python -m dataset_neuma.mains.aggregate_participants \
        --processed-dir /path/al/repo/dataset_neuma/data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def concat_pattern(dir_path: Path, pattern: str) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(dir_path.glob(pattern)):
        frames.append(pd.read_csv(csv_path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agrega perfiles y productos de todos los participantes.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "processed",
        help="Directorio con los CSV por participante.",
    )
    args = parser.parse_args()

    processed_dir = args.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    profiles = concat_pattern(processed_dir, "*_profile_demographics.csv")
    products = concat_pattern(processed_dir, "*_products.csv")

    profiles_out = processed_dir / "profiles_all.csv"
    products_out = processed_dir / "products_all.csv"

    profiles.to_csv(profiles_out, index=False)
    products.to_csv(products_out, index=False)

    print(f"Guardado: {profiles_out} (filas: {len(profiles)})")
    print(f"Guardado: {products_out} (filas: {len(products)})")


if __name__ == "__main__":
    main()
