# Paquetes
import pandas as pd

# Funciones de limpieza
def convertir_timestamp(df):
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')
    if df_copy['timestamp'].isna().sum() > 0:
        print(f"Advertencia: {df_copy['timestamp'].isna().sum()} timestamps no pudieron convertirse")
    return df_copy

def convertir_columnas_binarias(df):
    df = df.copy()
    for col in df.columns:
        vals = df[col].dropna().unique()
        if len(vals) == 2:
            # Caso 1: booleanas verdaderas
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)

            # Caso 2: strings tipo 'True'/'False'
            elif set(map(str.lower, vals.astype(str))) <= {'true', 'false'}:
                df[col] = df[col].astype(str).str.lower().map({'true': 1, 'false': 0})

            # Caso 3: binarios tipo {0,1}
            elif set(vals) <= {0, 1}:
                continue

            # Caso 4: otros dos valores arbitrarios
            else:
                mapping = {vals[0]: 0, vals[1]: 1}
                df[col] = df[col].map(mapping)
    return df


def convertir_a_categorico(df, umbral_unicos=50):
    df = df.copy()
    for col in df.columns:
        if df[col].nunique() < umbral_unicos:
            df[col] = df[col].astype('category')
    return df

def categorias_a_str(df):
    df = df.copy()
    for col in df.select_dtypes(include=['category', 'object']).columns:
        df[col] = df[col].astype(str)
    return df

def limpiar_dataset(df):
    df = convertir_timestamp(df)
    df = convertir_columnas_binarias(df)
    df = categorias_a_str(df)
    df = convertir_a_categorico(df)
    df.fillna(0, inplace=True)
    return df