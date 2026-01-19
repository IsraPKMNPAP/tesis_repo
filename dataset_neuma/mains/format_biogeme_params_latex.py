from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def format_float(x) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Formatea biogeme_params.csv a tabla LaTeX.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/icl_v_biogeme/biogeme_params.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/icl_v_biogeme/biogeme_params_latex.txt"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    for col in ["beta", "std", "tstat", "pval"]:
        if col in df.columns:
            df[col] = df[col].apply(format_float)

    lines = []
    for _, row in df.iterrows():
        line = ",".join(str(v) for v in row.tolist())
        line = line.replace(",", " & ") + " \\\\"
        lines.append(line)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Guardado: {args.output}")


if __name__ == "__main__":
    main()
