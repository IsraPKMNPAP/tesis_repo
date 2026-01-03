from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
import torchaudio


def load_segment(path: Union[str, Path], target_sr: int, sr_cache: int) -> Optional[torch.Tensor]:
    p = Path(path)
    if not p.exists():
        # Intentar sin doble .pt
        if p.suffix == ".pt" and p.name.endswith(".pt.pt"):
            p = p.with_suffix("")
    if not p.exists():
        return None
    try:
        try:
            seg = torch.load(p, map_location="cpu", weights_only=True)
        except TypeError:
            seg = torch.load(p, map_location="cpu")
    except Exception:
        return None
    if isinstance(seg, list):
        seg = torch.tensor(seg, dtype=torch.float32)
    elif isinstance(seg, tuple):
        seg = torch.tensor(seg[0], dtype=torch.float32)
    elif isinstance(seg, torch.Tensor):
        seg = seg.float()
    else:
        return None
    if seg.dim() == 1:
        seg = seg.unsqueeze(0)
    elif seg.dim() == 2 and seg.size(0) != 1:
        seg = seg.mean(dim=0, keepdim=True)
    if sr_cache != target_sr:
        seg = torchaudio.functional.resample(seg, sr_cache, target_sr)
    return seg


def main():
    ap = argparse.ArgumentParser(description="Extrae embeddings wav2vec (sin finetuning) de segmentos precortados.")
    ap.add_argument("--pkl", type=str, required=True, help="Pickle con columna de audio cacheado")
    ap.add_argument("--audio-col", type=str, default="audio_cached_path", help="Columna con rutas a segmentos .pt")
    ap.add_argument("--out-col", type=str, default="wav2vec_emb_path", help="Columna de salida con ruta al embedding guardado")
    ap.add_argument("--output-dir", type=str, default="data/processed/embeddings_audio_wav2vec", help="Directorio donde guardar los embeddings")
    ap.add_argument("--window-col", type=str, default="tensor_path_id", help="Columna para nombrar archivos (fallback al índice si falta)")
    ap.add_argument("--sr-cache", type=int, default=16000, help="Sample rate de los segmentos cacheados")
    ap.add_argument("--bundle", type=str, default="WAV2VEC2_BASE", help="Bundle de torchaudio.pipelines")
    ap.add_argument("--out-pkl", type=str, default=None, help="Ruta de salida; default: agrega columna y sobreescribe")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--limit", type=int, default=None, help="Procesa solo los primeros N registros (debug)")
    ap.add_argument("--overwrite", action="store_true", help="Sobrescribe embeddings existentes")
    args = ap.parse_args()

    df = pd.read_pickle(args.pkl).reset_index(drop=True)
    if args.limit:
        df = df.iloc[: args.limit].copy()
    if args.audio_col not in df.columns:
        raise KeyError(f"No se encontró la columna {args.audio_col} en el pickle")

    bundle = getattr(torchaudio.pipelines, args.bundle)
    model = bundle.get_model().to(args.device).eval()
    sr_model = bundle.sample_rate
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths = []
    n_ok, n_fail = 0, 0
    for idx, path in enumerate(df[args.audio_col]):
        if pd.isna(path):
            out_paths.append(None)
            n_fail += 1
            continue
        seg = load_segment(path, target_sr=sr_model, sr_cache=args.sr_cache)
        if seg is None:
            out_paths.append(None)
            n_fail += 1
            continue
        # Determinar nombre de archivo
        base_name = None
        if args.window_col in df.columns and pd.notna(df.iloc[idx][args.window_col]):
            base_name = str(df.iloc[idx][args.window_col])
        else:
            base_name = f"idx_{idx}"
        out_path = out_dir / f"{base_name}.pt"
        if out_path.exists() and not args.overwrite:
            out_paths.append(str(out_path))
            n_ok += 1
            continue
        try:
            with torch.no_grad():
                feats = model.extract_features(seg.to(args.device))[0]
                z = feats.mean(dim=1).squeeze(0).cpu()
            torch.save(z, out_path)
            out_paths.append(str(out_path))
            n_ok += 1
        except Exception:
            out_paths.append(None)
            n_fail += 1

    df[args.out_col] = out_paths
    out_path = args.out_pkl or args.pkl
    df.to_pickle(out_path)
    print(f"Embeddings extraídos: ok={n_ok}, fallidos={n_fail}")
    print(f"Guardado pickle en: {out_path}")
    print(f"Directorios de embeddings: {out_dir}")


if __name__ == "__main__":
    main()
