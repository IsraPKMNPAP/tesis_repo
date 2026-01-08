from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class DeterministicICLV(nn.Module):
    """ICLV determinista (sin Monte Carlo).

    - LT = Gamma(OBS_LT)
    - Indicadores: I_hat = Lambda(LT)
    - Utilidad j: U_j = ASC_j + beta^T OBS_U_j + delta_j^T LT
    - Prob: softmax(U)
    """

    def __init__(
        self,
        dim_obs_lt: int,
        dim_obs_u: int,
        n_latent: int,
        n_indicators: int,
        n_choices: int,
        alpha: float = 1.0,
        delta_per_alt: bool = True,
        beta_per_alt: bool = False,
    ):
        super().__init__()
        self.n_choices = int(n_choices)
        self.n_indicators = int(n_indicators)
        self.alpha = float(alpha)
        self.beta_per_alt = bool(beta_per_alt)

        # Bloque estructural y de medición
        self.Gamma = nn.Linear(dim_obs_lt, n_latent)
        self.Lambda = nn.Linear(n_latent, n_indicators) if n_indicators > 0 else None

        # Bloque de utilidad
        if self.beta_per_alt:
            self.beta = nn.Parameter(torch.zeros(n_choices, dim_obs_u))
        else:
            self.beta = nn.Linear(dim_obs_u, 1, bias=False)
        if delta_per_alt:
            self.delta = nn.Parameter(torch.zeros(n_choices, n_latent))
        else:
            self.delta = nn.Parameter(torch.zeros(n_latent))
        self.ASC = nn.Parameter(torch.zeros(n_choices))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.Gamma.weight)
        if self.Gamma.bias is not None:
            nn.init.zeros_(self.Gamma.bias)
        if self.Lambda is not None:
            nn.init.xavier_uniform_(self.Lambda.weight)
            if self.Lambda.bias is not None:
                nn.init.zeros_(self.Lambda.bias)
        if isinstance(self.beta, nn.Linear):
            nn.init.xavier_uniform_(self.beta.weight)
        else:
            nn.init.xavier_uniform_(self.beta)
        nn.init.zeros_(self.ASC)
        nn.init.zeros_(self.delta)

    def compute_utilities(self, obs_u: torch.Tensor, LT: torch.Tensor) -> torch.Tensor:
        """V_nj = beta^T OBS_U_nj + delta_j^T LT_n + ASC_j."""
        if obs_u.dim() != 3:
            raise ValueError(f"Se espera obs_u con shape [B, J, dim_obs_u]; se recibio {obs_u.shape}")
        if self.beta_per_alt:
            # obs_u: [B, J, dim_obs_u], beta: [J, dim_obs_u]
            beta_term = (obs_u * self.beta.unsqueeze(0)).sum(-1)
        else:
            beta_term = self.beta(obs_u).squeeze(-1)  # [B, J]
        if self.delta.dim() == 2:
            delta_term = LT @ self.delta.t()  # [B, J]
        else:
            delta_term = (LT @ self.delta).unsqueeze(1).expand_as(beta_term)  # [B, J]
        asc_term = self.ASC.unsqueeze(0)  # [1, J]
        return beta_term + delta_term + asc_term

    def forward(
        self,
        obs_lt: torch.Tensor,
        obs_u: torch.Tensor,
        indicators: torch.Tensor,
        choice: torch.Tensor,
    ):
        LT = self.Gamma(obs_lt)  # [B, n_latent]
        I_hat = self.Lambda(LT) if (self.Lambda is not None and self.n_indicators > 0) else None

        V = self.compute_utilities(obs_u, LT)  # [B, J]
        logp = F.log_softmax(V, dim=1)

        loss_choice = F.nll_loss(logp, choice, reduction="mean")
        if I_hat is None:
            loss_meas = torch.tensor(0.0, device=obs_lt.device, dtype=loss_choice.dtype)
        else:
            loss_meas = F.mse_loss(I_hat, indicators, reduction="mean")

        loss = loss_choice + self.alpha * loss_meas
        ll = logp.gather(1, choice.view(-1, 1)).sum()
        return {
            "loss": loss,
            "logp": logp,
            "LT": LT,
            "I_hat": I_hat,
            "loss_choice": loss_choice,
            "loss_meas": loss_meas,
            "log_likelihood": ll,
        }


@dataclass
class HessianResult:
    theta: torch.Tensor
    std: torch.Tensor
    tstat: torch.Tensor
    hessian: torch.Tensor
    var_covar: torch.Tensor
    names: List[str]


def trainable_named_params(model: nn.Module) -> List[tuple[str, nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def param_names(model: nn.Module) -> List[str]:
    names: List[str] = []
    for n, p in trainable_named_params(model):
        if p.numel() == 1:
            names.append(n)
        else:
            names.extend([f"{n}[{i}]" for i in range(p.numel())])
    return names


def compute_hessian_stats(model: nn.Module, loss_closure: Callable[[], torch.Tensor]) -> HessianResult:
    """Calcula Hessiano numérico y errores estándar a partir de una loss closure."""
    named_params = trainable_named_params(model)
    params: Sequence[nn.Parameter] = [p for _, p in named_params]
    names = param_names(model)

    flat_init = parameters_to_vector(params).detach()

    def _wrapped_loss(flat_params: torch.Tensor) -> torch.Tensor:
        vector_to_parameters(flat_params, params)
        return loss_closure()

    H = torch.autograd.functional.hessian(_wrapped_loss, flat_init)
    # Restaurar parámetros
    vector_to_parameters(flat_init, params)

    eye = torch.eye(H.shape[0], device=H.device, dtype=H.dtype) * 1e-4
    H_safe = H + eye
    try:
        H_inv = torch.linalg.pinv(H_safe)
    except Exception:
        H_inv = torch.linalg.pinv(H_safe + 1e-3 * eye)

    var = torch.diag(H_inv)
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    theta = flat_init
    tstat = theta / torch.clamp(std, min=1e-12)

    return HessianResult(
        theta=theta.detach(),
        std=std.detach(),
        tstat=tstat.detach(),
        hessian=H.detach(),
        var_covar=H_inv.detach(),
        names=names,
    )
