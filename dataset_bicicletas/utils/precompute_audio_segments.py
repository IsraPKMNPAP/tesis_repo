from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm


def load_wave(
    path: Path,
    target_sr: int,
) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(path))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


def extract_segment(
    waveform: torch.Tensor,
    start_s: float,
    duration_s: float,
    sr: int,
    normalize: bool,
) -> torch.Tensor:
    start = max(0, int(start_s * sr))
    length = max(int(duration_s * sr), 1)
    end = start + length
    if start >= waveform.size(-1):
        # devuelve silencio del largo solicitado
        segment = torch.zeros(1, length)
    else:
        segment = waveform[..., start:end]
        if segment.size(-1) < length:
            pad = length - segment.size(-1)
            segment = torch.nn.functional.pad(segment, (0, pad))
    if normalize:
        mean = segment.mean(dim=-1, keepdim=True)
        std = segment.std(dim=-1, keepdim=True) + 1e-6
        segment = (segment - mean) / std
    return segment


def main():
    ap = argparse.ArgumentParser(
        description="Pre-corta segmentos de audio y guarda rutas a tensores .pt para acelerar el pipeline multimodal."
    )
    ap.add_argument("--pkl", type=str, default="data/processed/multimodal_av_join.pkl", help="Pickle multimodal de entrada")
    ap.add_argument("--participant-col", type=str, default="participant")
    ap.add_argument("--start-col", type=str, default="audio_segment_start")
    ap.add_argument("--audio-col", type=str, default=None, help="Columna con ruta directa al audio (opcional)")
    ap.add_argument(
        "--audio-root",
        type=str,
        default="/mnt/otra_particion/home/israel_gpu_data/audio_data_raw/audio_participantes_validos",
        help="Raíz donde viven los raw_audio_<PARTICIPANTE>.wav",
    )
    ap.add_argument("--audio-template", type=str, default="raw_audio_{participant}.wav")
    ap.add_argument("--audio-patched-template", type=str, default="raw_audio_{participant}_patched.wav", help="Patrón alternativo; se prioriza si existe")
    ap.add_argument("--sr", type=int, default=8000, help="Sample rate de salida para los segmentos")
    ap.add_argument("--duration", type=float, default=5.0, help="Duración en segundos de cada segmento")
    ap.add_argument("--norm", action="store_true", help="Normaliza por canal cada segmento")
    ap.add_argument("--output-dir", type=str, default="/mnt/otra_particion/home/israel_gpu_data/audio_segments", help="Directorio donde guardar los .pt")
    ap.add_argument(
        "--output-pkl",
        type=str,
        default="data/processed/multimodal_av_join_audio_cached.pkl",
        help="Ruta de salida del pickle con columna audio_cached_path",
    )
    ap.add_argument("--overwrite", action="store_true", help="Sobrescribe .pt existentes")
    ap.add_argument("--limit", type=int, default=None, help="Solo procesa los primeros N registros (debug)")
    ap.add_argument("--window-col", type=str, default="tensor_path_id", help="Columna usada para nombrar archivos")
    args = ap.parse_args()

    df = pd.read_pickle(args.pkl).reset_index(drop=True)
    if args.limit:
        df = df.iloc[: args.limit].copy()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # caches
    wav_cache: Dict[str, torch.Tensor] = {}
    seg_cache: Dict[Tuple[str, float], Path] = {}

    audio_paths = []
    root = Path(args.audio_root) if args.audio_root else None

    def resolve_wav_path(participant: str) -> Optional[Path]:
        # Prioridad: patched -> template -> audio_col (cuando se llama aparte)
        if root:
            patched = root / args.audio_patched_template.format(participant=participant)
            if patched.exists():
                return patched
            normal = root / args.audio_template.format(participant=participant)
            if normal.exists():
                return normal
        return None

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Precomputando audio"):
        part = str(row.get(args.participant_col, ""))
        start_s = float(row.get(args.start_col, 0.0) or 0.0)
        wav_path = None
        if args.audio_col and args.audio_col in row and pd.notna(row[args.audio_col]):
            wav_path = Path(str(row[args.audio_col]))
        elif part:
            wav_path = resolve_wav_path(part)
        if wav_path is None or not wav_path.exists():
            audio_paths.append(None)
            continue

        cache_key = str(wav_path)
        if cache_key not in wav_cache:
            try:
                wav_cache[cache_key] = load_wave(wav_path, args.sr)
            except Exception:
                wav_cache[cache_key] = None
        waveform = wav_cache.get(cache_key)
        if waveform is None:
            audio_paths.append(None)
            continue

        seg_key = (cache_key, start_s)
        if seg_key in seg_cache and not args.overwrite:
            audio_paths.append(str(seg_cache[seg_key]))
            continue

        try:
            seg = extract_segment(waveform, start_s=start_s, duration_s=args.duration, sr=args.sr, normalize=args.norm)
            # Nombre de archivo: usa window_col si existe, si no índice
            base_name = f"audio_{row.get(args.window_col, f'idx_{idx}')}"
            out_path = out_dir / f"{base_name}.pt"
            torch.save(seg, out_path)
            seg_cache[seg_key] = out_path
            audio_paths.append(str(out_path))
        except Exception:
            audio_paths.append(None)

    df["audio_cached_path"] = audio_paths
    df.to_pickle(args.output_pkl)
    print(f"Guardado pickle con audio precortado en: {args.output_pkl}")
    print(f"Segmentos guardados en: {out_dir}")


if __name__ == "__main__":
    main()
