from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from .multimodal import MultimodalDataset, MultimodalSample  # reuse tab+video structure


@dataclass
class MultimodalAudioSample(MultimodalSample):
    x_aud: Optional[torch.Tensor] = None


def load_audio_segment(
    root: Path,
    template: str,
    participant: str,
    start_seconds: float,
    target_sr: int = 16000,
    duration_seconds: float = 5.0,
    normalize: bool = True,
    norm_mode: str = "per_channel",
    fallback_template: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """Carga un segmento fijo desde el raw_audio_{participant}.wav."""
    cand_templates = [template] + ([fallback_template] if fallback_template else [])
    wav_path = None
    for tmpl in cand_templates:
        p = root / tmpl.format(participant=participant)
        if p.exists():
            wav_path = p
            break
    if wav_path is None:
        return None
    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    start = max(0, int(start_seconds * sr))
    end = start + int(duration_seconds * sr)
    if start >= waveform.size(-1):
        return None
    segment = waveform[..., start:end]
    if segment.size(-1) < int(duration_seconds * sr):
        pad_len = int(duration_seconds * sr) - segment.size(-1)
        segment = torch.nn.functional.pad(segment, (0, pad_len))
    if normalize and norm_mode == "per_channel":
        mean = segment.mean(dim=-1, keepdim=True)
        std = segment.std(dim=-1, keepdim=True) + 1e-6
        segment = (segment - mean) / std
    return segment  # [1, T]


class MultimodalAudioDataset(Dataset):
    """Extiende MultimodalDataset para incluir audio opcional."""

    def __init__(
        self,
        df: pd.DataFrame,
        tab_columns: Sequence[str],
        X_tab_array: Optional[torch.Tensor] = None,
        path_col: str = "gpu_tensor_path",
        audio_col: Optional[str] = "audio_path",
        label_col: Optional[str] = None,
        timestamp_col: Optional[str] = "timestamp",
        window_id_col: Optional[str] = "window",
        participant_col: str = "participant",
        audio_start_col: str = "audio_segment_start",
        audio_root: Optional[str] = None,
        audio_template: str = "raw_audio_{participant}.wav",
        audio_fallback_template: Optional[str] = None,
        prefer_df_label: bool = True,
        class_map: Optional[dict] = None,
        video_transform=None,
        audio_sr: int = 16000,
        audio_duration: float = 5.0,
        audio_norm: str = "per_channel",
    ):
        self.df = df.reset_index(drop=True)
        self.tab_columns = list(tab_columns)
        self.X_tab_array = X_tab_array
        self.audio_col = audio_col if (audio_col and audio_col in df.columns) else None
        self.participant_col = participant_col
        self.audio_start_col = audio_start_col
        self.audio_root = Path(audio_root) if audio_root else None
        self.audio_template = audio_template
        self.audio_fallback_template = audio_fallback_template
        self.audio_sr = int(audio_sr)
        self.audio_duration = float(audio_duration)
        self.audio_norm = audio_norm
        # Reuse video dataset logic by composition
        self.inner = MultimodalDataset(
            df=self.df,
            tab_columns=self.tab_columns,
            X_tab_array=self.X_tab_array,
            path_col=path_col,
            label_col=label_col,
            timestamp_col=timestamp_col,
            window_id_col=window_id_col,
            prefer_df_label=prefer_df_label,
            class_map=class_map,
            video_transform=video_transform,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> MultimodalAudioSample:
        base = self.inner[idx]
        x_aud = None
        # Preferir audio_col directo; si no existe, usar raw + segment_start
        if self.audio_col and self.audio_col in self.df.columns and pd.notna(self.df.iloc[idx][self.audio_col]):
            try:
                x_aud = load_audio_segment(
                    root=Path(os.path.dirname(str(self.df.iloc[idx][self.audio_col]))),
                    template=os.path.basename(str(self.df.iloc[idx][self.audio_col])),
                    participant=str(self.df.iloc[idx].get(self.participant_col, "")),
                    start_seconds=float(self.df.iloc[idx].get(self.audio_start_col, 0.0) or 0.0),
                    target_sr=self.audio_sr,
                    duration_seconds=self.audio_duration,
                    normalize=True,
                    norm_mode=self.audio_norm,
                )
            except Exception:
                x_aud = None
        elif self.audio_root and self.participant_col in self.df.columns and self.audio_start_col in self.df.columns:
            part = str(self.df.iloc[idx][self.participant_col])
            start_s = float(self.df.iloc[idx][self.audio_start_col] or 0.0)
            x_aud = load_audio_segment(
                root=self.audio_root,
                template=self.audio_template,
                participant=part,
                start_seconds=start_s,
                target_sr=self.audio_sr,
                duration_seconds=self.audio_duration,
                normalize=True,
                norm_mode=self.audio_norm,
                fallback_template=self.audio_fallback_template,
            )
        return MultimodalAudioSample(
            x_tab=base.x_tab,
            x_vid=base.x_vid,
            x_aud=x_aud,
            y=base.y,
            timestamp=base.timestamp,
            window_id=base.window_id,
            participant=base.participant,
        )


def collate_multimodal_audio(batch: List[MultimodalAudioSample]):
    x_tabs = [b.x_tab for b in batch]
    x_vids = [b.x_vid for b in batch]
    x_auds = [b.x_aud for b in batch]
    ys = []
    ts, wids, parts = [], [], []

    def _coerce_label(v):
        if v is None:
            return -1
        try:
            if isinstance(v, torch.Tensor):
                return int(v.item())
            return int(v)
        except Exception:
            return -1

    for b in batch:
        ys.append(_coerce_label(b.y))
        ts.append(b.timestamp)
        wids.append(b.window_id)
        parts.append(b.participant)

    X_tab = torch.stack(x_tabs, dim=0)
    # Video stacking
    if x_vids[0].dim() in (3, 4):
        X_vid = torch.stack(x_vids, dim=0)
    else:
        raise ValueError(f"Dimensión de video no soportada: {x_vids[0].shape}")

    # Audio stacking: permitir None -> zero tensor
    has_audio = any(x is not None for x in x_auds)
    X_aud = None
    if has_audio:
        max_len = max(x.shape[-1] for x in x_auds if x is not None)
        aud_tensors = []
        for xa in x_auds:
            if xa is None:
                aud_tensors.append(torch.zeros(1, max_len))
            else:
                if xa.shape[-1] < max_len:
                    pad_len = max_len - xa.shape[-1]
                    xa = torch.nn.functional.pad(xa, (0, pad_len))
                elif xa.shape[-1] > max_len:
                    xa = xa[..., :max_len]
                aud_tensors.append(xa)
        X_aud = torch.stack(aud_tensors, dim=0)  # [B, 1, T]

    y = torch.tensor(ys, dtype=torch.long)

    class B:
        def __init__(self, x_tab, x_vid, x_aud, y, timestamp, window_id, participant):
            self.x_tab = x_tab
            self.x_vid = x_vid
            self.x_aud = x_aud
            self.y = y
            self.timestamp = timestamp
            self.window_id = window_id
            self.participant = participant

    return B(x_tab=X_tab, x_vid=X_vid, x_aud=X_aud, y=y, timestamp=ts, window_id=wids, participant=parts)
