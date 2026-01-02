from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalICLVDeterministic(nn.Module):
    """
    ICLV multimodal:
      - Gamma: LT = f([obs_lt, img_proj])
      - Medición: EEG_hat = Lambda(LT) (MSE con embedding EEG)
      - Utilidad: U_j = ASC_j + beta^T obs_u_j + delta_j^T LT
    """

    def __init__(
        self,
        dim_obs_lt: int,
        dim_obs_u: int,
        dim_img_emb: int,
        dim_eeg_emb: int,
        n_latent: int,
        n_choices: int,
        alpha: float = 1.0,
        delta_per_alt: bool = True,
        img_proj_dim: int = 32,
    ):
        super().__init__()
        self.n_choices = int(n_choices)
        self.alpha = float(alpha)
        self.n_latent = n_latent
        self.dim_eeg_emb = dim_eeg_emb

        self.img_proj = nn.Sequential(
            nn.Linear(dim_img_emb, img_proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        enc_in = dim_obs_lt + img_proj_dim
        self.Gamma = nn.Sequential(
            nn.Linear(enc_in, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_latent),
        )
        self.Lambda = (
            nn.Sequential(
                nn.Linear(n_latent, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, dim_eeg_emb),
            )
            if dim_eeg_emb > 0
            else None
        )

        self.beta = nn.Linear(dim_obs_u, 1, bias=False) if dim_obs_u > 0 else None
        if delta_per_alt:
            self.delta = nn.Parameter(torch.zeros(n_choices, n_latent))
        else:
            self.delta = nn.Parameter(torch.zeros(n_latent))
        self.ASC = nn.Parameter(torch.zeros(n_choices))

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.ASC)
        nn.init.zeros_(self.delta)

    def compute_utilities(self, obs_u: torch.Tensor, LT: torch.Tensor) -> torch.Tensor:
        if obs_u.dim() != 3:
            raise ValueError(f"Se espera obs_u [B, J, dim_obs_u]; got {obs_u.shape}")
        if self.beta is not None:
            beta_term = self.beta(obs_u).squeeze(-1)  # [B, J]
        else:
            beta_term = torch.zeros((obs_u.shape[0], obs_u.shape[1]), device=obs_u.device, dtype=obs_u.dtype)
        if self.delta.dim() == 2:
            delta_term = LT @ self.delta.t()  # [B, J]
        else:
            delta_term = (LT @ self.delta).unsqueeze(1).expand_as(beta_term)
        asc_term = self.ASC.unsqueeze(0)
        return beta_term + delta_term + asc_term

    def forward(self, obs_lt: torch.Tensor, obs_u: torch.Tensor, eeg_emb: torch.Tensor, img_emb: torch.Tensor, choice: torch.Tensor):
        img_p = self.img_proj(img_emb)
        LT = self.Gamma(torch.cat([obs_lt, img_p], dim=-1))
        eeg_hat = self.Lambda(LT) if self.Lambda is not None else None

        V = self.compute_utilities(obs_u, LT)
        logp = F.log_softmax(V, dim=1)

        loss_choice = F.nll_loss(logp, choice, reduction="mean")
        if eeg_hat is None:
            loss_meas = torch.tensor(0.0, device=obs_lt.device, dtype=loss_choice.dtype)
        else:
            loss_meas = F.mse_loss(eeg_hat, eeg_emb, reduction="mean")
        loss = loss_choice + self.alpha * loss_meas
        ll = logp.gather(1, choice.view(-1, 1)).sum()

        return {
            "loss": loss,
            "loss_choice": loss_choice,
            "loss_meas": loss_meas,
            "logp": logp,
            "LT": LT,
            "eeg_hat": eeg_hat,
            "log_likelihood": ll,
        }
