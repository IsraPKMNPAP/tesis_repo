from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

import pandas as pd
import torch
import open_clip
from PIL import Image


def load_tensor_or_image(path: Union[str, Path]) -> torch.Tensor:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        img = Image.open(p).convert("RGB")
        return img
    # Torch tensor
    t = torch.load(p, map_location="cpu")
    if isinstance(t, dict):
        # intentar claves comunes
        for key in ("frames", "video", "x"):
            if key in t:
                t = t[key]
                break
    if isinstance(t, torch.Tensor):
        if t.dim() == 4:  # [T,C,H,W]
            t = t[0]
        elif t.dim() == 3:
            pass
        else:
            raise ValueError(f"Tensor con dim no soportada: {t.shape}")
        # convertir a PIL
        if t.max() <= 1.0:
            t = (t * 255.0).clamp(0, 255)
        t = t.byte()
        img = Image.fromarray(t.permute(1, 2, 0).cpu().numpy())
        return img
    raise ValueError(f"Formato no soportado en {p}")


def main():
    ap = argparse.ArgumentParser(description="Extrae embeddings CLIP (sin finetuning) de frames/tensores de video.")
    ap.add_argument("--pkl", type=str, required=True, help="Pickle multimodal con columna de video")
    ap.add_argument("--path-col", type=str, default="frames_route", help="Columna con rutas a tensores/frames")
    ap.add_argument("--out-col", type=str, default="clip_emb_path", help="Columna de salida con rutas a embeddings")
    ap.add_argument("--output-dir", type=str, default="data/processed/embeddings_video_clip", help="Directorio para embeddings")
    ap.add_argument("--window-col", type=str, default="window", help="Columna para nombrar archivos (fallback al indice)")
    ap.add_argument("--model", type=str, default="RN50", help="Modelo CLIP de open_clip")
    ap.add_argument("--pretrained", type=str, default="openai", help="Pesos preentrenados")
    ap.add_argument("--out-pkl", type=str, default=None, help="Ruta de salida del pickle")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--log-file", type=str, default=None)
    args = ap.parse_args()

    df = pd.read_pickle(args.pkl).reset_index(drop=True)
    if args.limit:
        df = df.iloc[: args.limit].copy()
    if args.path_col not in df.columns:
        raise KeyError(f"No se encontró {args.path_col} en el pickle")

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=args.device)
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths = []
    errors = []
    n_ok = n_fail = 0
    for idx, path in enumerate(df[args.path_col]):
        if pd.isna(path):
            out_paths.append(None)
            n_fail += 1
            errors.append((idx, "nan path"))
            continue
        try:
            img = load_tensor_or_image(path)
            image = preprocess(img).unsqueeze(0).to(args.device)
            with torch.no_grad():
                z = model.encode_image(image).squeeze(0).cpu()
            base_name = None
            if args.window_col in df.columns and pd.notna(df.iloc[idx][args.window_col]):
                base_name = str(df.iloc[idx][args.window_col])
            else:
                base_name = f"{idx}"
            base_name = base_name.replace(".pt", "")
            out_path = out_dir / f"emb_window_{base_name}.pt"
            if out_path.exists() and not args.overwrite:
                out_paths.append(str(out_path))
                n_ok += 1
                continue
            torch.save(z, out_path)
            out_paths.append(str(out_path))
            n_ok += 1
        except Exception as e:
            out_paths.append(None)
            n_fail += 1
            errors.append((idx, str(e)))

    df[args.out_col] = out_paths
    out_pkl = args.out_pkl or args.pkl
    df.to_pickle(out_pkl)
    print(f"Embeddings CLIP extraidos: ok={n_ok}, fallidos={n_fail}")
    print(f"Guardado pickle en: {out_pkl}")
    print(f"Embeddings en: {out_dir}")
    if args.verbose and errors:
        print("Primeros errores:")
        for i, (idx, reason) in enumerate(errors[:10]):
            print(f"[{i}] idx={idx} reason={reason}")
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            for idx, reason in errors:
                f.write(f"{idx}\t{reason}\n")
        print(f"Log de errores guardado en: {log_path}")


if __name__ == "__main__":
    main()
