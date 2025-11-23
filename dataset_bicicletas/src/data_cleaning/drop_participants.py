#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if path.suffix in {".csv", ".tsv"}:
        return pd.read_csv(path)
    raise ValueError(f"Formato no soportado: {path}")


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in {".pkl", ".pickle"}:
        df.to_pickle(path)
    elif path.suffix in {".csv", ".tsv"}:
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Formato no soportado al guardar: {path}")


def drop_participants(df: pd.DataFrame, participants: List[str], participant_col: str) -> pd.DataFrame:
    parts_str = [str(p) for p in participants]
    mask = ~df[participant_col].astype(str).isin(parts_str)
    return df[mask].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Elimina participantes de pickles/CSV por lista de IDs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Rutas a pickles/CSV a procesar (se leen en bloque).")
    parser.add_argument("--participants", nargs="+", required=True, help="IDs de participantes a eliminar (ej: P21).")
    parser.add_argument("--participant-col", default="participant", help="Nombre de la columna de participante.")
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Sobrescribe cada archivo de entrada. Si se omite, escribe <nombre>_nopart.pkl junto al original.",
    )
    args = parser.parse_args()

    for inp in args.inputs:
        src = Path(inp)
        if not src.exists():
            print(f"[skip] No existe: {src}")
            continue
        try:
            df = load_table(src)
        except Exception as exc:
            print(f"[error] No se pudo leer {src}: {exc}")
            continue
        if args.participant_col not in df.columns:
            print(f"[skip] {src}: no tiene columna {args.participant_col}")
            continue

        before = len(df)
        df_out = drop_participants(df, args.participants, args.participant_col)
        removed = before - len(df_out)

        if args.inplace:
            dst = src
        else:
            dst = src.with_name(src.stem + "_nopart" + src.suffix)
        try:
            save_table(df_out, dst)
        except Exception as exc:
            print(f"[error] No se pudo guardar {dst}: {exc}")
            continue

        print(f"[ok] {src.name}: {removed} filas removidas ({len(df_out)} restantes) -> {dst}")


if __name__ == "__main__":
    main()
