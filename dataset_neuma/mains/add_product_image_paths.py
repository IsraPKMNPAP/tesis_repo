"""
Agrega la ruta de imagen recortada a `products_all.csv`.

Convención de imágenes (desde dataset_neuma):
  - PageY -> carpeta `data/processed/images/products/ImagePage_Y/`
  - ProductZ -> archivo `ImagePage_Y_crop_Z.png`

Ejemplo:
  Page1 + Product10 -> data/processed/images/products/ImagePage_1/ImagePage_1_crop_10.png

Uso (desde dataset_neuma):
  python -m dataset_neuma.mains.add_product_image_paths --input ./data/processed/products_all.csv --output ./data/processed/products_all_with_images.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Permite ejecución desde carpeta dataset_neuma
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))


PAGE_RE = re.compile(r"^Page(\d+)$", re.IGNORECASE)
PROD_RE = re.compile(r"^Product(\d+)$", re.IGNORECASE)


def to_image_path(page: str, product_id: str, base_dir: Path) -> Path:
    pm = PAGE_RE.match(str(page).strip())
    xm = PROD_RE.match(str(product_id).strip())
    if not pm or not xm:
        return Path("")
    page_n = int(pm.group(1))
    prod_n = int(xm.group(1))
    img_dir = base_dir / f"ImagePage_{page_n}"
    return img_dir / f"ImagePage_{page_n}_crop_{prod_n}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Agrega paths de imágenes a products_all.csv.")
    parser.add_argument("--input", type=Path, default=Path("./data/processed/products_all.csv"))
    parser.add_argument("--output", type=Path, default=Path("./data/processed/products_all_with_images.csv"))
    parser.add_argument(
        "--images-base",
        type=Path,
        default=Path("./data/processed/images/products"),
        help="Carpeta base donde están los recortes por ImagePage_X.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"No se encontró {args.input}")

    df = pd.read_csv(args.input)
    required = {"subject", "page", "product_id", "bought"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Faltan columnas requeridas en products_all.csv: {required - set(df.columns)}")

    paths = []
    exists = []
    for _, r in df.iterrows():
        p = to_image_path(r["page"], r["product_id"], args.images_base)
        paths.append(str(p))
        exists.append(bool(str(p)) and Path(p).exists())

    # Guardar como ruta relativa (desde dataset_neuma) si el usuario mantiene el cwd ahí
    df["image_path"] = paths
    df["image_exists"] = exists

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Guardado: {args.output} (filas: {len(df)}; image_exists True: {int(df['image_exists'].sum())})")


if __name__ == "__main__":
    main()
