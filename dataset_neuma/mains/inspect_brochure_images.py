"""
Aplica bounding boxes de folletos y recorta productos.

Uso (desde dataset_neuma):
  python -m dataset_neuma.mains.inspect_brochure_images ^
    --images-dir /mnt/otra_particion/home/israel_gpu_data/dataset_neuma/raw/Dependencies/Brochure_Pages ^
    --bboxes-dir /mnt/otra_particion/home/israel_gpu_data/dataset_neuma/raw/Dependencies/BoundingBox_Coordinates ^
    --menus-dir ./data/processed/images/menus ^
    --products-dir ./data/processed/images/products
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat
from scipy.io.matlab import mat_struct

# Permite ejecución desde carpeta dataset_neuma
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))


# ------------------------- utilidades de display -------------------------
def describe_value(val: Any) -> str:
    try:
        if isinstance(val, mat_struct):
            return f"mat_struct fields={getattr(val, '_fieldnames', [])}"
        if isinstance(val, np.ndarray):
            return f"ndarray dtype={val.dtype} shape={val.shape}"
        if isinstance(val, dict):
            return f"dict keys={list(val.keys())[:5]}"
        return repr(val)
    except Exception as exc:  # pragma: no cover - defensivo
        return f"(unprintable: {exc})"


def summarize_dict(d: Dict[str, Any], max_items: int = 20) -> None:
    keys = [k for k in d.keys() if not k.startswith("__")]
    if not keys:
        print("  (no public keys)")
    for k in keys[:max_items]:
        print(f"  - {k}: {describe_value(d[k])}")
    if len(keys) > max_items:
        print(f"  ... {len(keys) - max_items} more keys not shown")


# ------------------------- carga de .mat -------------------------
def load_bbox_mat(path: Path) -> Dict[str, Any]:
    data = loadmat(path, struct_as_record=False, squeeze_me=True)
    return {k: v for k, v in data.items() if not k.startswith("__")}


def _as_numeric_array(val: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(val)
    except Exception:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    return arr


# ------------------------- extracción de cajas -------------------------
def coerce_box_from_array(arr: np.ndarray) -> Optional[np.ndarray]:
    """Intenta obtener (x, y, w, h) de un array plano o 1x4/4x1/2x2."""
    flat = arr.reshape(-1)
    if flat.size < 4:
        return None
    vals = flat.astype(float)
    x1, y1, x2, y2 = vals[0], vals[1], vals[2], vals[3]
    if x2 >= x1 and y2 >= y1:
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)
    return vals[:4]


def extract_box_from_struct(obj: mat_struct) -> Optional[np.ndarray]:
    fields = getattr(obj, "_fieldnames", []) or []
    lower_fields = {f.lower(): f for f in fields}

    for candidate in ["boundingbox", "bbox", "rect", "roi"]:
        if candidate in lower_fields:
            raw = getattr(obj, lower_fields[candidate])
            arr = _as_numeric_array(raw)
            if arr is not None:
                box = coerce_box_from_array(arr)
                if box is not None:
                    return box

    combos = [
        ("x", "y", "w", "h"),
        ("x", "y", "width", "height"),
        ("left", "top", "right", "bottom"),
    ]
    for combo in combos:
        if all(c in lower_fields for c in combo):
            vals = [getattr(obj, lower_fields[c]) for c in combo]
            arrs = [_as_numeric_array(v) for v in vals]
            if any(a is None for a in arrs):
                continue
            nums = [float(a.reshape(-1)[0]) for a in arrs]
            if combo == ("left", "top", "right", "bottom"):
                x1, y1, x2, y2 = nums
                return np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)
            return np.array(nums[:4], dtype=float)
    return None


def extract_box_arrays(data: Dict[str, Any]) -> List[Tuple[str, np.ndarray]]:
    """Busca matrices numéricas Nx4 o listas de ROIs (objeto)."""
    candidates: List[Tuple[str, np.ndarray]] = []
    debug_msgs: List[str] = []

    for key, val in data.items():
        arr = _as_numeric_array(val)
        if arr is not None and arr.ndim == 2 and arr.shape[1] in (4, 5):
            boxes = arr[:, :4].astype(float)
            candidates.append((key, boxes))
            continue

        if isinstance(val, np.ndarray) and val.dtype == object:
            boxes: List[np.ndarray] = []
            for idx, item in enumerate(val.reshape(-1)):
                box = None
                if isinstance(item, mat_struct):
                    box = extract_box_from_struct(item)
                else:
                    arr_item = _as_numeric_array(item)
                    if arr_item is not None:
                        box = coerce_box_from_array(arr_item)
                if box is not None:
                    boxes.append(box)
                else:
                    debug_msgs.append(f"    ROI[{idx}] sin parsear: {describe_value(item)}")
            if boxes:
                candidates.append((key, np.vstack(boxes)))
            if debug_msgs:
                print("  Detalle de ROIs no parseados:")
                for msg in debug_msgs:
                    print(msg)
    return candidates


# ------------------------- dibujo y crops -------------------------
def draw_boxes(img: Image.Image, boxes: np.ndarray) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for (x, y, w, h) in boxes:
        rect = (x, y, x + w, y + h)
        draw.rectangle(rect, outline="red", width=3)
    return img


def save_crops(img: Image.Image, boxes: np.ndarray, out_dir: Path, stem: str) -> None:
    crops_dir = out_dir / stem
    crops_dir.mkdir(parents=True, exist_ok=True)
    for i, (x, y, w, h) in enumerate(boxes):
        rect = (x, y, x + w, y + h)
        crop = img.crop(rect)
        crop.save(crops_dir / f"{stem}_crop_{i+1}.png")


# ------------------------- flujo principal por imagen -------------------------
def find_bbox_file(img_path: Path, bboxes_dir: Path) -> Optional[Path]:
    stem = img_path.stem
    candidates = [stem + ".mat"]
    if stem.startswith("ImagePage_"):
        page_suffix = stem.replace("ImagePage_", "", 1)
        candidates.append(f"BoundingBoxPage_{page_suffix}.mat")
    for name in candidates:
        candidate_path = bboxes_dir / name
        if candidate_path.exists():
            return candidate_path
    print(f"  No se encontró .mat de cajas. Probados: {candidates}")
    return None


def process_image(
    img_path: Path,
    bboxes_dir: Optional[Path],
    menus_dir: Path,
    products_dir: Path,
    save_crops_flag: bool,
) -> None:
    print(f"\nImagen: {img_path.name}")
    menus_dir.mkdir(parents=True, exist_ok=True)
    products_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(img_path).convert("RGB")
    png_path = menus_dir / f"{img_path.stem}.png"
    image.save(png_path)
    print(f"  Menú PNG guardado en: {png_path}")

    if not bboxes_dir:
        return

    bbox_path = find_bbox_file(img_path, bboxes_dir)
    if not bbox_path:
        return

    print(f"  Leyendo cajas desde: {bbox_path}")
    bbox_data = load_bbox_mat(bbox_path)
    summarize_dict(bbox_data)

    candidates = extract_box_arrays(bbox_data)
    if not candidates:
        print("  No se detectaron cajas interpretables automáticamente. Muestra de arrays numéricos:")
        for key, val in bbox_data.items():
            arr = _as_numeric_array(val)
            if arr is not None:
                print(f"    * {key}: dtype={arr.dtype}, shape={arr.shape}, sample={arr.flatten()[:10].tolist()}")
        return

    key, boxes = candidates[0]
    print(f"  Usando '{key}' como cajas (asumido formato x, y, w, h)")
    overlay = draw_boxes(image.copy(), boxes)
    overlay_path = menus_dir / f"{img_path.stem}_overlay.png"
    overlay.save(overlay_path)
    print(f"  Overlay guardado en: {overlay_path}")

    if save_crops_flag:
        save_crops(image, boxes, products_dir, img_path.stem)
        print(f"  Recortes guardados en carpeta {products_dir / img_path.stem}")


# ------------------------- CLI -------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Aplica bounding boxes y recorta productos desde folletos.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/mnt/otra_particion/home/israel_gpu_data/dataset_neuma/raw/Dependencies/Brochure_Pages"),
        help="Directorio con imágenes .tif",
    )
    parser.add_argument(
        "--bboxes-dir",
        type=Path,
        default=Path("/mnt/otra_particion/home/israel_gpu_data/dataset_neuma/raw/Dependencies/BoundingBox_Coordinates"),
        help="Directorio con .mat de cajas",
    )
    parser.add_argument(
        "--menus-dir",
        type=Path,
        default=Path("./data/processed/images/menus"),
        help="Dónde guardar los menús (png y overlay)",
    )
    parser.add_argument(
        "--products-dir",
        type=Path,
        default=Path("./data/processed/images/products"),
        help="Dónde guardar los recortes de productos",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Guardar recortes individuales de cada caja (x, y, w, h).",
    )
    args = parser.parse_args()

    images_dir = args.images_dir
    if not images_dir.exists():
        raise SystemExit(f"Directorio de imágenes no encontrado: {images_dir}")
    bboxes_dir = args.bboxes_dir if args.bboxes_dir else None

    tif_files = sorted(images_dir.glob("*.tif"))
    if not tif_files:
        print("No se encontraron .tif en el directorio de imágenes.")
        return

    for img_path in tif_files:
        process_image(img_path, bboxes_dir, args.menus_dir, args.products_dir, args.save_crops)


if __name__ == "__main__":
    main()
