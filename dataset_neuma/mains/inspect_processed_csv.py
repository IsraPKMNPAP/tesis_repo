from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

# Ruta por defecto apuntando al almacenamiento del GPU.
DEFAULT_PROCESSED_ROOT = Path("/mnt/otra_particion/home/israel_gpu_data/dataset_neuma/processed")
DEFAULT_MAX_UNIQUE = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Carga un CSV desde la carpeta processed del dataset_neuma, "
            "imprime metadata básica y convierte a category las columnas con baja cardinalidad."
        )
    )
    parser.add_argument(
        "--csv-name",
        required=True,
        help="Nombre del archivo .csv dentro de la carpeta processed.",
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_PROCESSED_ROOT),
        help=f"Ruta a la carpeta processed (por defecto: {DEFAULT_PROCESSED_ROOT}).",
    )
    parser.add_argument(
        "--max-unique",
        type=int,
        default=DEFAULT_MAX_UNIQUE,
        help="Máximo de valores únicos para convertir una columna a category.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Cantidad de filas a mostrar del head.",
    )
    return parser.parse_args()


def load_dataframe(csv_name: str, data_root: Path) -> Tuple[Path, pd.DataFrame]:
    csv_path = data_root / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo solicitado: {csv_path}")
    df = pd.read_csv(csv_path)
    return csv_path, df


def convert_low_cardinality(df: pd.DataFrame, max_unique: int) -> Tuple[pd.DataFrame, List[str]]:
    """Convierte a category las columnas con menos de max_unique valores únicos."""
    converted_cols: List[str] = []
    df_processed = df.copy()
    for col in df_processed.columns:
        uniques = df_processed[col].nunique(dropna=True)
        if uniques <= max_unique:
            df_processed[col] = df_processed[col].astype("category")
            converted_cols.append(col)
    return df_processed, converted_cols


def collect_dtypes(columns: Iterable[Tuple[str, str]], target_dtype: str) -> List[str]:
    return [col for col, dtype in columns if dtype == target_dtype]


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser()

    csv_path, df = load_dataframe(args.csv_name, data_root)
    print(f"Dataset: {csv_path.name}")
    print(f"Ruta: {csv_path}")
    print(f"Forma: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Columnas ({len(df.columns)}): {list(df.columns)}\n")

    print(f"Head (primeras {args.head} filas):")
    print(df.head(args.head))
    print(f"\nPreprocesando columnas con <= {args.max_unique} valores únicos...")

    df_processed, converted_cols = convert_low_cardinality(df, args.max_unique)
    print(f"Columnas convertidas a category ({len(converted_cols)}): {converted_cols}\n")

    dtypes = list(df_processed.dtypes.items())
    float_cols = collect_dtypes(dtypes, "float64")
    int_cols = collect_dtypes(dtypes, "int64")
    category_cols = collect_dtypes(dtypes, "category")

    print("Columnas float64:", float_cols)
    print("Columnas int64:", int_cols)
    print("Columnas category:", category_cols)


if __name__ == "__main__":
    main()
