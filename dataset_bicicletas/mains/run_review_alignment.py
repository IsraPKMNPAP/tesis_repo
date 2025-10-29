from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys
import pandas as pd

# Ensure package root on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p))


def _to_dt(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    if out[col].isna().any():
        n = int(out[col].isna().sum())
        raise ValueError(f"{n} valores inválidos en '{col}' (no convertibles a datetime)")
    return out


def main():
    ap = argparse.ArgumentParser(description="Revisión posterior a la alineación vs pickle ancla")
    ap.add_argument("--csv-aligned", type=str, required=True, help="CSV ya alineado al pickle")
    ap.add_argument("--pkl-ref", type=str, required=True, help="Pickle ancla con timestamps e is_imputed")
    ap.add_argument("--timestamp-col", type=str, default="timestamp")
    ap.add_argument("--participant-col", type=str, default="participant")
    ap.add_argument("--session-id-col", type=str, default="session_id")
    ap.add_argument("--imputed-col", type=str, default="is_imputed")
    ap.add_argument("--expected-step-seconds", type=int, default=5)
    args = ap.parse_args()

    csv_p = _expand(args.csv_aligned)
    ref_p = _expand(args.pkl_ref)
    if not csv_p.exists():
        raise FileNotFoundError(f"No existe CSV alineado: {csv_p}")
    if not ref_p.exists():
        raise FileNotFoundError(f"No existe Pickle referencia: {ref_p}")

    df_csv = pd.read_csv(csv_p)
    df_ref = pd.read_pickle(ref_p)

    ts = args.timestamp_col
    part = args.participant_col
    sess = args.session_id_col
    imp = args.imputed_col

    df_csv = _to_dt(df_csv, ts)
    df_ref = _to_dt(df_ref, ts)

    # 1) Conteo y cobertura
    print("=== Tamaños ===")
    print(f"CSV alineado: {len(df_csv)} | Pickle ancla: {len(df_ref)}")

    # 2) Chequeo 5s por grupo
    print("\n=== Chequeo Δt=5s por (participante, sesión) ===")
    viol_total = 0
    for gkey, g in df_csv.groupby([part, sess], dropna=False):
        gg = g.sort_values(ts)
        deltas = gg[ts].diff().dropna()
        bad = deltas.dt.total_seconds() != args.expected_step_seconds
        n_bad = int(bad.sum())
        if n_bad > 0:
            viol_total += n_bad
            print(f" - Grupo {gkey}: {n_bad} violaciones")
    if viol_total == 0:
        print("OK: todos los pasos de tiempo son de 5s")

    # 3) Cobertura de timestamps: todos los del pickle deben existir en CSV alineado
    print("\n=== Cobertura de timestamps (pickle ⊆ csv) ===")
    k_ref = df_ref[[part, sess, ts]].drop_duplicates()
    k_csv = df_csv[[part, sess, ts]].drop_duplicates()
    merged = k_ref.merge(k_csv, on=[part, sess, ts], how="left", indicator=True)
    missing = merged[merged["_merge"] == "left_only"]
    if len(missing) == 0:
        print("OK: todos los timestamps del pickle están en el CSV alineado")
    else:
        print(f"WARN: {len(missing)} timestamps del pickle faltan en el CSV alineado")

    # 4) Donde pickle tiene is_imputed=1, CSV debería tener is_imputed=1 y fila repetida de la previa (columna ts cambia)
    print("\n=== Chequeo correspondencia de imputaciones ===")
    if imp in df_ref.columns:
        ref_imp = df_ref[df_ref[imp] == 1][[part, sess, ts]].copy()
        chk = ref_imp.merge(df_csv[[part, sess, ts, imp]], on=[part, sess, ts], how="left")
        n_bad_flag = int((chk[imp] != 1).fillna(True).sum())
        if n_bad_flag == 0:
            print("OK: flags is_imputed=1 del pickle replicados en CSV")
        else:
            print(f"WARN: {n_bad_flag} timestamps imputados en pickle no tienen is_imputed=1 en CSV")
    else:
        print("(pickle no tiene columna is_imputed; se omite este chequeo)")

    print("\nRevisión completa.")


if __name__ == "__main__":
    main()

