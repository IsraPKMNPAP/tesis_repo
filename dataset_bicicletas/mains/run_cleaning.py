import sys
import os
import argparse
import pandas as pd

# Permitir import de src/* al ejecutar desde dataset_bicicletas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_cleaning.cleaning import limpiar_dataset


def main():
    parser = argparse.ArgumentParser(description="Limpia un CSV y lo guarda procesado")
    parser.add_argument("--csv-in", required=True, help="Ruta al CSV de entrada (requerido)")
    parser.add_argument("--csv-out", required=True, help="Ruta al CSV de salida (requerido)")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.csv_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Leyendo datos desde: {args.csv_in}")
    df = pd.read_csv(args.csv_in)
    print(f"Filas originales: {len(df)}")

    df_clean = limpiar_dataset(df)

    df_clean.to_csv(args.csv_out, index=False)
    print(f"Archivo limpio guardado en: {args.csv_out}")
    print(f"Filas finales: {len(df_clean)}")


if __name__ == "__main__":
    main()

