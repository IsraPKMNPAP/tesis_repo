from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class MultimodalICLVDataset(Dataset):
    """
    Dataset para ICLV multimodal:
      - obs_lt (tabular) + proyección de imagen se usan para generar LT
      - obs_u para utilidad
      - indicadores: embeddings EEG (target de reconstrucción)
      - choice: etiqueta (int)
    """

    def __init__(
        self,
        df,
        obs_lt_cols: Sequence[str],
        obs_u_cols: Sequence[str],
        label_col: str,
        img_emb_col: str,
        eeg_emb_col: str,
        num_choices: int = 2,
    ):
        self.df = df.reset_index(drop=True)
        self.obs_lt_cols = list(obs_lt_cols)
        self.obs_u_cols = list(obs_u_cols)
        self.label_col = label_col
        self.img_emb_col = img_emb_col
        self.eeg_emb_col = eeg_emb_col
        self.num_choices = int(num_choices)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        obs_lt = torch.tensor(row[self.obs_lt_cols].to_numpy(dtype=np.float32))
        obs_u = torch.tensor(row[self.obs_u_cols].to_numpy(dtype=np.float32))
        # repetir obs_u para cada alternativa
        obs_u_rep = obs_u.unsqueeze(0).repeat(self.num_choices, 1)  # [J, dim]
        choice = int(row[self.label_col])

        img_emb = np.load(row[self.img_emb_col]).astype(np.float32)
        if img_emb.ndim > 1:
            img_emb = img_emb.flatten()
        img_emb_t = torch.tensor(img_emb)

        eeg_emb = np.load(row[self.eeg_emb_col]).astype(np.float32)
        if eeg_emb.ndim > 1:
            eeg_emb = eeg_emb.flatten()
        eeg_emb_t = torch.tensor(eeg_emb)

        return obs_lt, obs_u_rep, eeg_emb_t, img_emb_t, choice


def collate_fn(batch):
    obs_lt, obs_u, eeg_emb, img_emb, choice = zip(*batch)
    obs_lt_t = torch.stack(obs_lt, dim=0)
    obs_u_t = torch.stack(obs_u, dim=0)
    eeg_t = torch.stack(eeg_emb, dim=0)
    img_t = torch.stack(img_emb, dim=0)
    choice_t = torch.tensor(choice, dtype=torch.long)
    return obs_lt_t, obs_u_t, eeg_t, img_t, choice_t
