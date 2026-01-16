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
        delta_per_alt: bool = True,
        beta_per_alt: bool = False,
        img_proj_dim: int = 32,
    ):
        super().__init__()
        self.n_choices = int(n_choices)
        self.n_latent = n_latent
        self.beta_per_alt = bool(beta_per_alt)
        self.dim_eeg_emb = dim_eeg_emb
        self.base_alt = 0
        jm1 = self.n_choices - 1

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

        if dim_obs_u > 0:
            if self.beta_per_alt:
                self.beta = nn.Parameter(torch.zeros(jm1, dim_obs_u))
            else:
                self.beta = nn.Linear(dim_obs_u, 1, bias=False)
        else:
            self.beta = None
        if delta_per_alt:
            self.delta = nn.Parameter(torch.zeros(jm1, n_latent))
        else:
            self.delta = nn.Parameter(torch.zeros(n_latent))
        self.ASC = nn.Parameter(torch.zeros(jm1))

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
        B, J, _ = obs_u.shape
        device = obs_u.device
        dtype = obs_u.dtype
        asc_full = torch.zeros(J, device=device, dtype=dtype)
        if self.ASC.numel() > 0:
            asc_full[1:] = self.ASC
        if self.beta is None:
            beta_term = torch.zeros((obs_u.shape[0], obs_u.shape[1]), device=obs_u.device, dtype=obs_u.dtype)
        else:
            if self.beta_per_alt:
                beta_full = torch.zeros(J, obs_u.size(-1), device=device, dtype=dtype)
                beta_full[1:, :] = self.beta
                beta_term = (obs_u * beta_full.unsqueeze(0)).sum(-1)
            else:
                beta_term = self.beta(obs_u).squeeze(-1)
        if self.delta.dim() == 2:
            delta_full = torch.zeros(J, LT.size(-1), device=device, dtype=dtype)
            delta_full[1:, :] = self.delta
            delta_term = LT @ delta_full.t()  # [B, J]
        else:
            delta_term = (LT @ self.delta).unsqueeze(1).expand_as(beta_term)
        asc_term = asc_full.unsqueeze(0)
        return beta_term + delta_term + asc_term

    def forward(self, obs_lt: torch.Tensor, obs_u: torch.Tensor, eeg_emb: torch.Tensor, img_emb: torch.Tensor, choice: torch.Tensor):
        img_p = self.img_proj(img_emb)
        LT = self.Gamma(torch.cat([obs_lt, img_p], dim=-1))
        if self.Lambda is not None and self.Lambda[0].weight.numel() > 0:
            with torch.no_grad():
                self.Lambda[0].weight[0, 0] = 1.0
        eeg_hat = self.Lambda(LT) if self.Lambda is not None else None

        V = self.compute_utilities(obs_u, LT)
        logp = F.log_softmax(V, dim=1)

        ll_choice = logp.gather(1, choice.view(-1, 1)).sum()
        if eeg_hat is None:
            ll_meas = torch.tensor(0.0, device=obs_lt.device, dtype=ll_choice.dtype)
        else:
            ll_meas = -0.5 * torch.pow(eeg_hat - eeg_emb, 2).sum()
        total_loglik = ll_choice + ll_meas
        loss_choice = -ll_choice
        loss_meas = -ll_meas
        loss = -total_loglik

        return {
            "loss": loss,
            "loss_choice": loss_choice,
            "loss_meas": loss_meas,
            "logp": logp,
            "LT": LT,
            "eeg_hat": eeg_hat,
            "log_likelihood": total_loglik,
            "loglik_choice_sum": ll_choice,
            "loglik_meas_sum": ll_meas,
        }
