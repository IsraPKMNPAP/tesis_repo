from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Tuple

import pandas as pd

# Ensure package root on path (to be consistent with other mains/* scripts)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p))


def _coerce_ts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"No se encontró la columna de timestamp '{col}' en el DataFrame")
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    n_na = int(out[col].isna().sum())
    if n_na > 0:
        raise ValueError(
            f"{n_na} valores de '{col}' no pudieron convertirse a datetime. Revise los datos antes del join."
        )
    return out


def _check_uniqueness(df: pd.DataFrame, col: str, name: str) -> int:
    dup = int(df.duplicated(subset=[col]).sum())
    if dup > 0:
        print(f"[WARN] {dup} timestamps duplicados en {name} para la columna '{col}'.")
    return dup


def _validate_sets(
    a: pd.Series, b: pd.Series, label_a: str, label_b: str, sample: int = 5
) -> Tuple[bool, list[str]]:
    sa, sb = set(a.tolist()), set(b.tolist())
    mism_a = sa - sb
    mism_b = sb - sa
    msgs: list[str] = []
    ok = True
    if mism_a:
        ok = False
        ex = list(sorted(mism_a))[:sample]
        msgs.append(
            f"{len(mism_a)} timestamps de {label_a} no están en {label_b}. Ejemplos: {ex}"
        )
    if mism_b:
        ok = False
        ex = list(sorted(mism_b))[:sample]
        msgs.append(
            f"{len(mism_b)} timestamps de {label_b} no están en {label_a}. Ejemplos: {ex}"
        )
    return ok, msgs


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Join tabular CSV y pickle de video por timestamp, con validaciones estrictas por defecto."
        )
    )
    ap.add_argument(
        "--csv-in",
        type=str,
        default="~/projects/tesis_repo/dataset_bicicletas/data/processed/dataset_bicicletas_clean.csv",
        help="Ruta al CSV tabular limpio",
    )
    ap.add_argument(
        "--pkl-in",
        type=str,
        default="~/projects/tesis_repo/dataset_bicicletas/data/processed/X_proc_final_linked.pkl",
        help="Ruta al pickle con paths linkeados a GPU",
    )
    ap.add_argument(
        "--timestamp-col",
        type=str,
        default="timestamp",
        help="Nombre de la columna de timestamp a usar en ambos archivos",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="~/projects/tesis_repo/dataset_bicicletas/data/processed/multimodal_join.pkl",
        help="Ruta de salida (pickle) para el DataFrame combinado",
    )
    ap.add_argument(
        "--how",
        type=str,
        choices=["one-to-one", "inner", "left", "right", "outer"],
        default="one-to-one",
        help=(
            "Estrategia de merge: 'one-to-one' valida 1:1 y mismas claves. "
            "Use 'inner' para intersección si hay pequeñas discrepancias."
        ),
    )
    ap.add_argument(
        "--suffixes",
        type=str,
        nargs=2,
        default=("_csv", "_vid"),
        help="Sufijos para columnas en conflicto (par de strings)",
    )
    args = ap.parse_args()

    csv_path = _expand(args.csv_in)
    pkl_path = _expand(args.pkl_in)
    out_path = _expand(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el CSV: {csv_path}")
    if not pkl_path.exists():
        raise FileNotFoundError(f"No existe el pickle: {pkl_path}")

    print(f"Leyendo CSV: {csv_path}")
    df_csv = pd.read_csv(csv_path)
    print(f"Leyendo Pickle: {pkl_path}")
    df_vid = pd.read_pickle(pkl_path)

    # Coerción de timestamps
    ts_col = args.timestamp_col
    df_csv = _coerce_ts(df_csv, ts_col)
    df_vid = _coerce_ts(df_vid, ts_col)

    # Unicidad y tamaños
    n_dup_csv = _check_uniqueness(df_csv, ts_col, "CSV")
    n_dup_vid = _check_uniqueness(df_vid, ts_col, "Pickle")
    print(
        f"Filas CSV={len(df_csv)} | Pickle={len(df_vid)} | Duplicados CSV={n_dup_csv} | Duplicados Pickle={n_dup_vid}"
    )

    # Validación de sets si user quiere 1:1
    validate_arg = None
    how = "inner" if args.how == "one-to-one" else args.how
    if args.how == "one-to-one":
        same_keys, msgs = _validate_sets(df_csv[ts_col], df_vid[ts_col], "CSV", "Pickle")
        if not same_keys:
            print("\n[ERROR] Las claves de timestamp no coinciden entre CSV y Pickle:")
            for m in msgs:
                print(" -", m)
            print(
                "\nSugerencia: vuelva a ejecutar con '--how inner' para conservar solo la intersección"
            )
            raise SystemExit(2)
        if n_dup_csv == 0 and n_dup_vid == 0:
            validate_arg = "one_to_one"
        elif n_dup_csv == 0 and n_dup_vid > 0:
            validate_arg = "one_to_many"
        elif n_dup_csv > 0 and n_dup_vid == 0:
            validate_arg = "many_to_one"
        else:
            validate_arg = None  # many_to_many, dejar que pandas lo maneje

    # Merge
    print(f"Realizando merge por '{ts_col}' con how='{how}' ...")
    merged = pd.merge(
        df_csv,
        df_vid,
        on=ts_col,
        how=how,
        validate=validate_arg,
        suffixes=tuple(args.suffixes),
        copy=False,
    )

    # Ordenar por timestamp para consistencia
    merged = merged.sort_values(by=ts_col).reset_index(drop=True)

    # Guardar en pickle
    merged.to_pickle(out_path)
    print(f"Guardado merge en: {out_path} | Filas: {len(merged)} | Columnas: {len(merged.columns)}")


if __name__ == "__main__":
    main()

