from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Lista variables categoricas y sus valores unicos.")
    ap.add_argument("--data", type=str, required=True, help="CSV o PKL con el dataset.")
    ap.add_argument("--max-unique", type=int, default=10, help="Umbral para categoricas.")
    ap.add_argument("--cols-file", type=str, default=None, help="Archivo con columnas a revisar (txt).")
    ap.add_argument("--sample", type=int, default=0, help="Muestreo aleatorio de filas (0=todo).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    path = Path(args.data)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
    else:
        df = pd.read_pickle(path)

    if args.sample and args.sample > 0:
        df = df.sample(n=min(args.sample, len(df)), random_state=args.seed).reset_index(drop=True)

    cols = None
    if args.cols_file:
        cols = [c.strip() for c in Path(args.cols_file).read_text(encoding="utf-8").splitlines() if c.strip()]
        cols = [c for c in cols if c in df.columns]
    if cols is None:
        cols = df.columns.tolist()

    print(f"Total columnas evaluadas: {len(cols)}")
    for col in cols:
        try:
            nunique = df[col].nunique(dropna=True)
        except Exception:
            continue
        if nunique <= args.max_unique:
            vals = df[col].dropna().astype(str).unique().tolist()
            print(f"{col} (n_unique={nunique}): {vals}")


if __name__ == "__main__":
    main()
