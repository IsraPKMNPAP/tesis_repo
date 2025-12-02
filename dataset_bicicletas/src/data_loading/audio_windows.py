from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.functional as F


def format_participant_token(value, prefix: str = "P", zero_pad: int = 2) -> str:
    raw = str(value).strip().upper()
    if raw.startswith(prefix.upper()):
        digits = "".join(ch for ch in raw if ch.isdigit())
        if digits:
            return f"{prefix}{int(digits):0{zero_pad}d}"
        return raw
    digits = "".join(ch for ch in raw if ch.isdigit())
    if digits:
        return f"{prefix}{int(digits):0{zero_pad}d}"
    raise ValueError(f"No se pudo normalizar participante: {value}")


@dataclass
class AudioMeta:
    path: Path
    sample_rate: int
    num_frames: int
    duration: float


class AudioSegmentDataset(Dataset):
    """Dataset que recorta ventanas de audio (5s) usando frame_offset/num_frames."""

    def __init__(
        self,
        df: pd.DataFrame,
        audio_root: str | Path,
        participant_col: str = "participant",
        start_col: str = "audio_segment_start",
        label_col: str = "label_id",
        timestamp_col: str | None = "timestamp",
        window_seconds: float = 5.0,
        sample_rate: int = 16000,
        filename_template: str = "raw_audio_{participant}.wav",
        participant_prefix: str = "P",
        participant_zero_pad: int = 2,
        strict: bool = True,
        cache_audio: bool = False,
        aug_noise_std: float = 0.0,
        aug_prob: float = 0.0,
        audio_cached_col: str | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.audio_root = Path(audio_root) if audio_root else None
        self.participant_col = participant_col
        self.start_col = start_col
        self.label_col = label_col
        self.timestamp_col = timestamp_col if timestamp_col in self.df.columns else None
        self.window_seconds = window_seconds
        self.sample_rate = sample_rate
        self.filename_template = filename_template
        self.participant_prefix = participant_prefix
        self.participant_zero_pad = participant_zero_pad
        self.strict = strict
        self.cache_audio = cache_audio
        self.aug_noise_std = aug_noise_std
        self.aug_prob = aug_prob
        self.audio_cached_col = audio_cached_col

        self.participant_tokens: Dict[str, str] = {}
        self.audio_meta: Dict[str, AudioMeta] = {}
        self.audio_cache: Dict[str, torch.Tensor] = {}
        self._prepare_audio_index()

    def _prepare_audio_index(self) -> None:
        # Si hay audio precortado en cache, podemos saltar la preparación pesada
        has_cached_col = self.audio_cached_col and self.audio_cached_col in self.df.columns
        has_cached_values = bool(has_cached_col and self.df[self.audio_cached_col].notna().any())
        if self.audio_root is None:
            if has_cached_values:
                return
            if self.strict:
                raise FileNotFoundError(
                    "audio_root es requerido cuando no hay audio precortado. "
                    "Si usas cache, asegúrate de que la columna audio_cached_path tenga valores."
                )
            return

        unique_parts = self.df[self.participant_col].astype(str).unique()
        for value in unique_parts:
            token = format_participant_token(value, self.participant_prefix, self.participant_zero_pad)
            path = self.audio_root / self.filename_template.format(participant=token)
            if not path.exists():
                if self.strict:
                    raise FileNotFoundError(f"No existe el audio requerido para {value}: {path}")
                continue
            info = torchaudio.info(str(path))
            sr = info.sample_rate or self.sample_rate
            duration = info.num_frames / sr if sr else 0.0
            self.participant_tokens[str(value)] = token
            self.audio_meta[str(value)] = AudioMeta(path=path, sample_rate=sr, num_frames=info.num_frames, duration=duration)
            if self.cache_audio:
                waveform, sr_loaded = torchaudio.load(str(path))
                if sr_loaded != self.sample_rate:
                    waveform = F.resample(waveform, sr_loaded, self.sample_rate)
                self.audio_cache[str(value)] = waveform.to(torch.float32)
        if self.strict and len(self.participant_tokens) != len(unique_parts):
            missing = set(map(str, unique_parts)) - set(self.participant_tokens.keys())
            raise FileNotFoundError(f"No se encontró audio para participantes: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def _slice_waveform(self, participant_key: str, start_seconds: float) -> torch.Tensor:
        meta = self.audio_meta[participant_key]
        frame_offset = int(round(start_seconds * meta.sample_rate))
        num_frames = int(round(self.window_seconds * meta.sample_rate))
        frame_offset = max(frame_offset, 0)
        num_frames = max(num_frames, 1)

        if self.cache_audio and participant_key in self.audio_cache:
            full = self.audio_cache[participant_key]
            end = frame_offset + num_frames
            slice_wave = full[:, frame_offset:end]
            if slice_wave.size(-1) < num_frames:
                pad_frames = num_frames - slice_wave.size(-1)
                slice_wave = torch.nn.functional.pad(slice_wave, (0, pad_frames))
            waveform = slice_wave
            sr = self.sample_rate
        else:
            waveform, sr = torchaudio.load(str(meta.path), frame_offset=frame_offset, num_frames=num_frames)
            needed_frames = int(round(self.window_seconds * sr))
            if waveform.size(-1) < needed_frames:
                pad_frames = needed_frames - waveform.size(-1)
                waveform = torch.nn.functional.pad(waveform, (0, pad_frames))
            if sr != self.sample_rate:
                waveform = F.resample(waveform, sr, self.sample_rate)
        return waveform.to(torch.float32)

    def _maybe_augment(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.aug_noise_std > 0 and self.aug_prob > 0:
            if torch.rand(1).item() < self.aug_prob:
                noise = torch.randn_like(waveform) * self.aug_noise_std
                waveform = waveform + noise
        return waveform

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        participant_key = str(row[self.participant_col])
        start = float(row[self.start_col])

        waveform = None
        # Caso 0: audio precalculado
        if self.audio_cached_col and self.audio_cached_col in row and pd.notna(row[self.audio_cached_col]):
            path = Path(str(row[self.audio_cached_col]))
            if path.exists():
                seg = None
                # Intento seguro usando weights_only (evita código arbitrario)
                try:
                    seg = torch.load(path, map_location="cpu", weights_only=True)
                except TypeError:
                    # weights_only no soportado en torch anterior: fallback
                    seg = torch.load(path, map_location="cpu")
                except Exception:
                    seg = None
                if seg is not None:
                    if seg.dim() == 1:
                        seg = seg.unsqueeze(0)
                    elif seg.dim() == 2 and seg.size(0) != 1:
                        seg = seg.mean(dim=0, keepdim=True)
                    target_frames = max(int(round(self.window_seconds * self.sample_rate)), 1)
                    if seg.size(-1) < target_frames:
                        seg = torch.nn.functional.pad(seg, (0, target_frames - seg.size(-1)))
                    elif seg.size(-1) > target_frames:
                        seg = seg[..., :target_frames]
                    waveform = seg.to(torch.float32)

        if waveform is None:
            if participant_key not in self.participant_tokens:
                raise KeyError(f"Sin audio para {participant_key} y sin cache disponible")
            waveform = self._slice_waveform(participant_key, start)
        waveform = self._maybe_augment(waveform)
        label = int(row[self.label_col])
        sample = {
            "waveform": waveform,
            "label": label,
            "participant": participant_key,
        }
        if self.timestamp_col:
            ts_val = row[self.timestamp_col]
            if isinstance(ts_val, pd.Timestamp):
                ts_val = ts_val.isoformat()
            elif pd.isna(ts_val):
                ts_val = None
            else:
                ts_val = str(ts_val)
            sample["timestamp"] = ts_val
        return sample


def split_by_participant(
    df: pd.DataFrame,
    participant_col: str,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    participants = df[participant_col].astype(str).unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(participants)
    val_size = max(1, int(round(len(participants) * val_ratio)))
    val_parts = set(participants[:val_size])
    mask = df[participant_col].astype(str).isin(val_parts)
    train_df = df[~mask].reset_index(drop=True)
    val_df = df[mask].reset_index(drop=True)
    return train_df, val_df


def create_audio_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    dataset_kwargs: Dict[str, object],
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = AudioSegmentDataset(train_df, **dataset_kwargs)
    val_dataset = AudioSegmentDataset(val_df, **dataset_kwargs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader
