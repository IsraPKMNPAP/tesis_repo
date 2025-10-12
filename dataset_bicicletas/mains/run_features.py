import sys
import os
import argparse
from pathlib import Path
import pandas as pd

# Permitir imports desde utils/* al ejecutar desde dataset_bicicletas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.features import (
    apply_feature_diff,
    feature_hash,
    format_columns_paged,
    save_features_file,
)


def read_columns(csv_path: str | Path) -> list[str]:
    # Leer solo encabezados para ser rápido
    df0 = pd.read_csv(csv_path, nrows=0)
    return df0.columns.tolist()


def prompt_loop(all_cols: list[str], selected: list[str]) -> list[str]:
    print("Modo interactivo. Comandos: add <col1,col2>, remove <col1,col2>, show, done")
    current = selected
    all_set = set(all_cols)
    while True:
        cmd = input("> ").strip()
        if not cmd:
            continue
        if cmd == "done":
            return current
        if cmd == "show":
            print(f"Seleccionadas ({len(current)}):")
            print("\n".join(current))
            continue
        if cmd.startswith("add "):
            names = [c.strip() for c in cmd[4:].split(",") if c.strip()]
            current = apply_feature_diff(all_cols, base=current, add=names)
            print(f"Añadidas: {names}")
            continue
        if cmd.startswith("remove "):
            names = [c.strip() for c in cmd[7:].split(",") if c.strip()]
            current = apply_feature_diff(all_cols, base=current, remove=names)
            print(f"Removidas: {names}")
            continue
        print("Comando no reconocido.")


def main():
    ap = argparse.ArgumentParser(description="Herramienta para seleccionar features")
    ap.add_argument("--csv", required=True, help="Ruta al CSV base para listar columnas")
    ap.add_argument("--base", nargs="*", default=None, help="Lista base de columnas (opcional)")
    ap.add_argument("--add", nargs="*", default=None, help="Columnas a agregar")
    ap.add_argument("--remove", nargs="*", default=None, help="Columnas a remover")
    ap.add_argument("--page-size", type=int, default=10, help="Tamaño de página para imprimir columnas")
    ap.add_argument("--interactive", action="store_true", help="Activar modo interactivo")
    ap.add_argument("--save", type=str, default=None, help="Guardar features en .json o .txt")
    ap.add_argument("--print-cmd", action="store_true", help="Imprimir comando sugerido para run_training")
    ap.add_argument("--label", type=str, default="action", help="Label para comando sugerido")
    ap.add_argument("--no-clean", action="store_true", help="Usar --no-clean en comando sugerido")
    ap.add_argument("--mnlogit", action="store_true", help="Incluir --mnlogit en comando sugerido")
    ap.add_argument("--torch", action="store_true", help="Incluir --torch en comando sugerido")
    args = ap.parse_args()

    cols = read_columns(args.csv)
    print("Columnas del CSV (paginadas):")
    print(format_columns_paged(cols, page_size=args.page_size))

    selected = apply_feature_diff(cols, base=args.base, add=args.add, remove=args.remove)
    if args.interactive:
        selected = prompt_loop(cols, selected)

    print("\nSelección final:")
    print("\n".join(selected))
    print(f"Total: {len(selected)} | hash: {feature_hash(selected)}")

    saved_path = None
    if args.save:
        meta = {"csv": args.csv, "count": len(selected)}
        save_features_file(args.save, selected, meta=meta)
        saved_path = args.save
        print(f"Guardado en: {saved_path}")

    if args.print_cmd:
        base_cmd = [
            "python",
            "mains/run_training.py",
            "--csv",
            args.csv,
        ]
        if args.no_clean:
            base_cmd.append("--no-clean")
        if args.mnlogit:
            base_cmd.append("--mnlogit")
        if args.torch:
            base_cmd.append("--torch")
        # Preferir features-file si guardamos, para no romper por longitud
        if saved_path:
            base_cmd += ["--features-file", saved_path]
        else:
            base_cmd += ["--features"] + selected
        # Imprimir una sola línea
        print("\nComando sugerido:")
        print(" ".join(base_cmd))


if __name__ == "__main__":
    main()
