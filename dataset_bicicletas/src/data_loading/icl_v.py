from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset


class ICLVDataset(Dataset):
    """Dataset simple para ICLV determinista."""

    def __init__(
        self,
        obs_lt: Sequence,
        obs_u: Sequence,
        indicators: Optional[Sequence],
        choices: Sequence,
        num_choices: int,
    ):
        self.obs_lt = torch.as_tensor(obs_lt, dtype=torch.float32)
        obs_u_tensor = torch.as_tensor(obs_u, dtype=torch.float32)
        if obs_u_tensor.dim() == 2:
            obs_u_tensor = obs_u_tensor.unsqueeze(1).expand(-1, num_choices, -1)
        elif obs_u_tensor.dim() != 3:
            raise ValueError(f"obs_u debe tener 2 o 3 dims; se recibio {obs_u_tensor.shape}")
        self.obs_u = obs_u_tensor

        if indicators is None:
            self.indicators = torch.zeros((len(self.obs_lt), 0), dtype=torch.float32)
        else:
            ind_tensor = torch.as_tensor(indicators, dtype=torch.float32)
            if ind_tensor.dim() == 1:
                ind_tensor = ind_tensor.unsqueeze(1)
            self.indicators = ind_tensor

        self.choices = torch.as_tensor(choices, dtype=torch.long)
        self.num_choices = int(num_choices)

    def __len__(self) -> int:
        return len(self.obs_lt)

    def __getitem__(self, idx: int):
        return (
            self.obs_lt[idx],
            self.obs_u[idx],
            self.indicators[idx],
            int(self.choices[idx]),
        )
