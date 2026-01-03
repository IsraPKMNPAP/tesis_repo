from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


def guess_existing_path(p: Path) -> Optional[Path]:
    """Intenta resolver el archivo correcto manejando doble sufijo .pt."""
    # Caso: nombre termina en .pt.pt -> preferir quitar un sufijo
    if p.suffix == ".pt" and p.name.endswith(".pt.pt"):
        cand = p.with_suffix("")  # quita un .pt
        if cand.exists():
            return cand
    if p.exists():
        return p
    # Caso .pt.pt -> quitar un sufijo
    if p.suffix == ".pt" and p.name.endswith(".pt.pt"):
        cand = p.with_suffix("")
        if cand.exists():
            return cand
    # Caso falta un .pt extra
    cand2 = p.with_suffix(p.suffix + ".pt")
    if not p.name.endswith(".pt") and cand2.exists():
        return cand2
    # Caso: cualquier archivo con mismo stem en el directorio
    if p.parent.exists():
        matches: List[Path] = list(p.parent.glob(p.stem + "*.pt"))
        if matches:
            return matches[0]
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Repara rutas de audio_cached_path (ej. elimina doble .pt) y opcionalmente renombra archivos."
    )
    ap.add_argument("--pkl", type=str, required=True, help="Pickle con columna de segmentos precortados")
    ap.add_argument("--audio-col", type=str, default="audio_cached_path", help="Nombre de la columna con rutas .pt")
    ap.add_argument("--out-pkl", type=str, default=None, help="Ruta de salida; si no se da, sobreescribe el pickle de entrada")
    ap.add_argument("--rename-files", action="store_true", help="Renombra archivos al path corregido (si es necesario)")
    ap.add_argument("--dry-run", action="store_true", help="No escribe ni renombra; solo reporta")
    args = ap.parse_args()

    df = pd.read_pickle(args.pkl).reset_index(drop=True)
    col = args.audio_col
    if col not in df.columns:
        raise KeyError(f"No se encontró la columna {col} en el pickle")

    fixed_paths = []
    n_fixed, n_missing = 0, 0
    for raw in df[col]:
        if pd.isna(raw):
            fixed_paths.append(raw)
            continue
        p = Path(str(raw))
        # Si termina en .pt.pt, preferimos normalizar aunque exista
        target = p
        cand = guess_existing_path(p)
        if cand is None:
            n_missing += 1
            fixed_paths.append(raw)
            continue
        # Si el nombre original ya es "raro", proponer nombre limpio
        if p.suffix == ".pt" and p.name.endswith(".pt.pt"):
            target = p.with_suffix("")
        # Si no vamos a renombrar, igual guardamos la ruta encontrada/limpia
        if args.rename_files and not args.dry_run and cand != target:
            try:
                cand.rename(target)
                fixed_paths.append(str(target))
                n_fixed += 1
            except Exception:
                fixed_paths.append(str(cand))
                n_fixed += 1
        else:
            fixed_paths.append(str(target if cand.exists() else cand))
            if str(target) != str(raw):
                n_fixed += 1

    print(f"Entradas corregidas: {n_fixed}, faltantes: {n_missing}")
    if args.dry_run:
        print("Modo dry-run: no se guardó pickle ni se renombraron archivos.")
        return

    df[col] = fixed_paths
    out_path = args.out_pkl or args.pkl
    df.to_pickle(out_path)
    print(f"Guardado pickle con rutas corregidas en: {out_path}")


if __name__ == "__main__":
    main()
