from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Selecciona columnas por correlacion con etiqueta y baja colinealidad.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cols-file", type=Path, required=True)
    parser.add_argument("--label-col", type=str, default="bought")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-corr", type=float, default=0.7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.lower()
    label = args.label_col.lower()
    if label not in df.columns:
        raise ValueError(f"No se encontro etiqueta '{label}'.")

    cols = [c.strip().lower() for c in args.cols_file.read_text().splitlines() if c.strip()]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        raise ValueError("No se encontraron columnas en el dataset.")

    X = df[cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[label], errors="coerce")

    # corr con etiqueta
    corr_label = X.corrwith(y).abs().sort_values(ascending=False)

    selected = []
    for col in corr_label.index:
        if len(selected) >= args.top_k:
            break
        ok = True
        for s in selected:
            c = X[[col, s]].corr().iloc[0, 1]
            if np.isnan(c):
                continue
            if abs(c) >= args.max_corr:
                ok = False
                break
        if ok:
            selected.append(col)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(f"Selected ({len(selected)}): {selected}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
