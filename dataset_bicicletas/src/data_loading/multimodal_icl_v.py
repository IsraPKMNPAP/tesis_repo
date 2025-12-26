from __future__ import annotations

from typing import List, Sequence, Optional

import torch
from torch.utils.data import Dataset

from .multimodal_audio import MultimodalAudioDataset


class MultimodalICLVDataset(Dataset):
    """Wrap de MultimodalAudioDataset para exponer obs_u e indicadores."""

    def __init__(
        self,
        base_mm: MultimodalAudioDataset,
        obs_u: torch.Tensor,
        indicators: torch.Tensor,
        n_choices: int,
    ):
        self.base = base_mm
        # Expandir obs_u por alternativa si no viene ya expandido
        if obs_u.dim() == 2:
            obs_u = obs_u.unsqueeze(1).expand(-1, n_choices, -1)
        self.obs_u = obs_u
        self.indicators = indicators
        self.n_choices = int(n_choices)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        return sample, self.obs_u[idx], self.indicators[idx]


def collate_multimodal_icl_v(batch: List):
    base_samples, obs_us, indicators = zip(*batch)
    # Reutilizar collate de audio para video/audio/tab
    from .multimodal_audio import collate_multimodal_audio

    collated = collate_multimodal_audio(base_samples)
    obs_u_tensor = torch.stack(list(obs_us), dim=0)
    indicators_tensor = torch.stack(list(indicators), dim=0)

    class B:
        def __init__(self, base, obs_u, indicators):
            self.x_tab = base.x_tab
            self.x_vid = base.x_vid
            self.x_aud = base.x_aud
            self.y = base.y
            self.timestamp = base.timestamp
            self.window_id = base.window_id
            self.participant = base.participant
            self.obs_u = obs_u
            self.indicators = indicators

    return B(collated, obs_u_tensor, indicators_tensor)
