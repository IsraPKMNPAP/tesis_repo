import sys
import os
import pandas as pd
# Añadir la raíz del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_cleaning.cleaning import limpiar_dataset


# Paths relativos al directorio raíz
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT_DIR, "data", "raw", "all_data.csv")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "dataset_bicicicletas_clean.csv")

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"Leyendo datos desde: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)

    print(f"Filas originales: {len(df)}")
    df_clean = limpiar_dataset(df)

    df_clean.to_csv(PROCESSED_PATH, index=False)
    print(f"Archivo limpio guardado en: {PROCESSED_PATH}")
    print(f"Filas finales: {len(df_clean)}")

if __name__ == "__main__":
    main()
