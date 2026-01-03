from __future__ import annotations

from typing import List, Sequence, Optional
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_emb(val):
    """Carga un embedding desde ruta .pt/.npy o retorna el tensor/array tal cual."""
    if isinstance(val, torch.Tensor):
        return val
    if isinstance(val, np.ndarray):
        return torch.tensor(val, dtype=torch.float32)
    if isinstance(val, (list, tuple)):
        return torch.tensor(np.asarray(val), dtype=torch.float32)
    # Interpretar como ruta
    p = Path(val)
    if p.suffix.lower() == ".pt":
        try:
            emb = torch.load(p, map_location="cpu", weights_only=True)
        except TypeError:
            emb = torch.load(p, map_location="cpu")
        if isinstance(emb, np.ndarray):
            emb = torch.tensor(emb, dtype=torch.float32)
        return emb.float()
    if p.suffix.lower() in [".npy", ".npz"]:
        arr = np.load(p)
        if isinstance(arr, np.lib.npyio.NpzFile):
            # tomar el primer array
            arr = arr[list(arr.files)[0]]
        return torch.tensor(arr, dtype=torch.float32)
    raise ValueError(f"No se pudo cargar embedding desde {val}")


class PrecomputedICLVDataset(Dataset):
    """Dataset que alimenta embeddings precalculados (video/audio) a ICLV."""

    def __init__(
        self,
        df,
        obs_lt_array: np.ndarray,
        obs_u_array: np.ndarray,
        indicator_array: np.ndarray,
        choice_array: np.ndarray,
        vid_emb_col: Optional[str] = None,
        aud_emb_col: Optional[str] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.obs_lt = torch.tensor(obs_lt_array, dtype=torch.float32)
        self.obs_u = torch.tensor(obs_u_array, dtype=torch.float32)
        self.indicators = torch.tensor(indicator_array, dtype=torch.float32)
        self.choice = torch.tensor(choice_array, dtype=torch.long)
        self.vid_emb_col = vid_emb_col
        self.aud_emb_col = aud_emb_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        vid_emb = None
        aud_emb = None
        if self.vid_emb_col and self.vid_emb_col in self.df.columns:
            vid_emb = _load_emb(self.df.iloc[idx][self.vid_emb_col])
        if self.aud_emb_col and self.aud_emb_col in self.df.columns:
            aud_emb = _load_emb(self.df.iloc[idx][self.aud_emb_col])
        return (
            self.obs_lt[idx],
            vid_emb,
            aud_emb,
            self.obs_u[idx],
            self.indicators[idx],
            int(self.choice[idx]),
        )


def collate_precomputed_icl_v(batch: List):
    obs_lt, vid_embs, aud_embs, obs_us, indicators, choices = zip(*batch)
    obs_lt_t = torch.stack(list(obs_lt), dim=0)
    obs_u_t = torch.stack(list(obs_us), dim=0)  # [B, dim_obs_u]
    indicators_t = torch.stack(list(indicators), dim=0)
    choice_t = torch.tensor(list(choices), dtype=torch.long)

    def pad_stack(emb_list):
        if all(e is None for e in emb_list):
            return None
        tensors = []
        max_dim = max(e.numel() for e in emb_list if e is not None)
        for e in emb_list:
            if e is None:
                tensors.append(torch.zeros(max_dim))
            else:
                flat = e.flatten().float()
                if flat.numel() < max_dim:
                    pad = torch.zeros(max_dim - flat.numel(), dtype=flat.dtype)
                    flat = torch.cat([flat, pad], dim=0)
                else:
                    flat = flat[:max_dim]
                tensors.append(flat)
        return torch.stack(tensors, dim=0)

    vid_t = pad_stack(vid_embs)
    aud_t = pad_stack(aud_embs)

    # Expandir obs_u a [B, J, dim_obs_u] si viene [B, dim_obs_u]
    if obs_u_t.dim() == 2:
        num_choices = int(choice_t.max().item() + 1) if choice_t.numel() > 0 else 1
        obs_u_t = obs_u_t.unsqueeze(1).expand(-1, num_choices, -1)

    class B:
        def __init__(self, x_tab, vid, aud, obs_u, indicators, y):
            self.x_tab = x_tab
            self.vid_emb = vid
            self.aud_emb = aud
            self.obs_u = obs_u
            self.indicators = indicators
            self.y = y

    return B(obs_lt_t, vid_t, aud_t, obs_u_t, indicators_t, choice_t)
