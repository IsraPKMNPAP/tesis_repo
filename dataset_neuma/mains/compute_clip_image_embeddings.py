"""
Genera embeddings de imágenes de productos usando CLIP congelado (sin fine-tuning).

Lee `products_all_with_images.csv`, toma productos únicos por (page, product_id),
calcula embedding CLIP y lo guarda en una carpeta por producto:
  data/processed/image_embeddings/{PageY}_{ProductZ}/embedding.npy
  data/processed/image_embeddings/{PageY}_{ProductZ}/meta.json

También escribe un índice:
  data/processed/image_embeddings/embeddings_index.csv

Uso (desde dataset_neuma):
  python -m dataset_neuma.mains.compute_clip_image_embeddings --products ./data/processed/products_all_with_images.csv --out-dir ./data/processed/image_embeddings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# open_clip_torch
import open_clip
import torch

# Permite ejecución desde carpeta dataset_neuma
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Embeddings CLIP de imágenes de productos (congelado).")
    parser.add_argument("--products", type=Path, default=Path("./data/processed/products_all_with_images.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("./data/processed/image_embeddings"))
    parser.add_argument("--model", type=str, default="ViT-B-32", help="Modelo CLIP (open_clip).")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k", help="Pesos preentrenados (open_clip).")
    parser.add_argument("--device", type=str, default="cuda", help="cuda o cpu.")
    parser.add_argument("--overwrite", action="store_true", help="Recalcular embeddings aunque existan.")
    args = parser.parse_args()

    if not args.products.exists():
        raise SystemExit(f"No se encontró {args.products}")

    df = pd.read_csv(args.products)
    required = {"page", "product_id", "image_path"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Faltan columnas requeridas en {args.products}: {required - set(df.columns)}")

    # Productos únicos (page, product_id) -> una sola imagen canónica
    df_unique = (
        df.dropna(subset=["image_path"])
        .drop_duplicates(subset=["page", "product_id"])
        .reset_index(drop=True)
    )
    if df_unique.empty:
        raise SystemExit("No hay filas con image_path para procesar.")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device)
    model.eval()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows: List[Dict[str, str]] = []

    with torch.inference_mode():
        for _, r in tqdm(df_unique.iterrows(), total=len(df_unique), desc="CLIP embeddings"):
            page = str(r["page"])
            product_id = str(r["product_id"])
            image_path = Path(str(r["image_path"]))

            key_dir = out_dir / f"{page}_{product_id}"
            emb_path = key_dir / "embedding.npy"
            meta_path = key_dir / "meta.json"

            if emb_path.exists() and (not args.overwrite):
                index_rows.append(
                    {
                        "page": page,
                        "product_id": product_id,
                        "image_path": str(image_path),
                        "embedding_path": str(emb_path),
                        "model": args.model,
                        "pretrained": args.pretrained,
                    }
                )
                continue

            if not image_path.exists():
                # si viene relativo, se evalúa desde cwd=dataset_neuma
                print(f"[WARN] Imagen no encontrada: {image_path} (skip)")
                continue

            key_dir.mkdir(parents=True, exist_ok=True)
            img = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(img).unsqueeze(0).to(device)
            features = model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            emb = features.squeeze(0).detach().cpu().numpy().astype(np.float32)

            np.save(emb_path, emb)
            meta = {
                "page": page,
                "product_id": product_id,
                "image_path": str(image_path),
                "model": args.model,
                "pretrained": args.pretrained,
                "embedding_dim": int(emb.shape[0]),
            }
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

            index_rows.append(
                {
                    "page": page,
                    "product_id": product_id,
                    "image_path": str(image_path),
                    "embedding_path": str(emb_path),
                    "model": args.model,
                    "pretrained": args.pretrained,
                }
            )

    index_df = pd.DataFrame(index_rows).sort_values(["page", "product_id"]).reset_index(drop=True)
    index_path = out_dir / "embeddings_index.csv"
    index_df.to_csv(index_path, index=False)
    print(f"Guardado: {index_path} (filas: {len(index_df)})")


if __name__ == "__main__":
    main()

