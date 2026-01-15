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
        if self.Lambda is not None and self.Lambda.weight.numel() > 0:
            with torch.no_grad():
                self.Lambda.weight[0, 0] = 1.0
        I_hat = self.Lambda(LT) if (self.Lambda is not None and self.n_indicators > 0) else None

        V = self.compute_utilities(obs_u, LT)  # [B, J]
        logp = F.log_softmax(V, dim=1)

        ll_choice = logp.gather(1, choice.view(-1, 1)).sum()
        if I_hat is None:
            ll_meas = torch.tensor(0.0, device=obs_lt.device, dtype=ll_choice.dtype)
        else:
            ll_meas = -0.5 * torch.pow(I_hat - indicators, 2).sum()
        total_loglik = ll_choice + ll_meas
        loss_choice = -ll_choice
        loss_meas = -ll_meas
        loss = -total_loglik
        return {
            "loss": loss,
            "logp": logp,
            "LT": LT,
            "I_hat": I_hat,
            "loss_choice": loss_choice,
            "loss_meas": loss_meas,
            "log_likelihood": total_loglik,
            "loglik_choice_sum": ll_choice,
            "loglik_meas_sum": ll_meas,
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


def compute_hessian_stats(
    model: nn.Module,
    loss_closure: Callable[[], torch.Tensor],
    n_samples: int | None = None,
    ridge: float = 1e-6,
) -> HessianResult:
    """Calcula Hessiano numérico y errores estándar a partir de una loss closure."""
    named_params = trainable_named_params(model)
    params: Sequence[nn.Parameter] = [p for _, p in named_params]
    names = param_names(model)

    flat_init = parameters_to_vector(params).detach()

    def _wrapped_loss(flat_params: torch.Tensor) -> torch.Tensor:
        vector_to_parameters(flat_params, params)
        return loss_closure()

    H = torch.autograd.functional.hessian(_wrapped_loss, flat_init)
    if n_samples is not None and n_samples > 0:
        H = H * float(n_samples)
    # Restaurar parámetros
    vector_to_parameters(flat_init, params)

    eye = torch.eye(H.shape[0], device=H.device, dtype=H.dtype) * float(ridge)
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


def compute_choice_hessian_stats_only_utility(
    model: nn.Module,
    batch: dict,
) -> HessianResult:
    """Hessiano/SE/t-stats solo para betas de utilidad usando -loglik (sum)."""
    util_params = []
    util_names: List[str] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("beta"):
            util_params.append(p)
            if p.numel() == 1:
                util_names.append(n)
            else:
                util_names.extend([f"{n}[{i}]" for i in range(p.numel())])

    flat_init = parameters_to_vector(util_params).detach()

    def _wrapped_nll(flat_params: torch.Tensor) -> torch.Tensor:
        vector_to_parameters(flat_params, util_params)
        out = model(
            obs_lt=batch["obs_lt"],
            obs_u=batch["obs_u"],
            indicators=batch.get("indicators"),
            choice=batch["choice"],
        )
        return -out["loglik_choice_sum"]

    H = torch.autograd.functional.hessian(_wrapped_nll, flat_init)
    vector_to_parameters(flat_init, util_params)
    eps = 1e-6
    eye = torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
    H_safe = H + eps * eye
    H_inv = torch.linalg.pinv(H_safe)

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
        names=util_names,
    )


def compute_biogeme_hessian_stats_full(
    model: nn.Module,
    batch: dict,
) -> tuple[HessianResult, dict, np.ndarray, np.ndarray]:
    """Biogeme/BHHH stats usando log-likelihood total (sum) y todos los params."""
    named_params = trainable_named_params(model)
    params: Sequence[nn.Parameter] = [p for _, p in named_params]
    names = param_names(model)

    flat_init = parameters_to_vector(params).detach()

    # gradients per observation (BHHH)
    obs_lt = batch["obs_lt"]
    obs_u = batch["obs_u"]
    indicators = batch.get("indicators")
    choice = batch["choice"]
    grads = []
    for i in range(obs_lt.shape[0]):
        out = model(
            obs_lt=obs_lt[i : i + 1],
            obs_u=obs_u[i : i + 1],
            indicators=indicators[i : i + 1] if indicators is not None else None,
            choice=choice[i : i + 1],
        )
        loglik_i = out["log_likelihood"]
        grad = torch.autograd.grad(loglik_i, params, retain_graph=False, create_graph=False)
        g = torch.cat([g_.reshape(-1) for g_ in grad]).detach().double().cpu().numpy()
        grads.append(g)
    G = np.stack(grads, axis=0) if grads else np.zeros((0, flat_init.numel()))
    B = G.T @ G if G.size else np.zeros((flat_init.numel(), flat_init.numel()))

    def _wrapped_nll(flat_params: torch.Tensor) -> torch.Tensor:
        vector_to_parameters(flat_params, params)
        out = model(obs_lt=obs_lt, obs_u=obs_u, indicators=indicators, choice=choice)
        return -out["log_likelihood"]

    H = torch.autograd.functional.hessian(_wrapped_nll, flat_init)
    vector_to_parameters(flat_init, params)
    A = H.detach().double().cpu().numpy()
    eigvals = np.linalg.eigvalsh(-A) if A.size else np.array([])
    diag = {
        "lambda_min": float(np.min(eigvals)) if eigvals.size else np.nan,
        "lambda_max": float(np.max(eigvals)) if eigvals.size else np.nan,
        "cond": float(np.max(eigvals) / np.min(eigvals)) if eigvals.size and np.min(eigvals) != 0 else np.inf,
        "n_obs": int(obs_lt.shape[0]),
    }
    invA = np.linalg.pinv(-A) if A.size else np.zeros_like(A)
    cov_classic = invA
    std_classic = np.sqrt(np.clip(np.diag(cov_classic), 1e-12, None))
    t_classic = flat_init.detach().double().cpu().numpy() / std_classic
    cov_robust = invA @ B @ invA if B.size else np.zeros_like(invA)
    std_robust = np.sqrt(np.clip(np.diag(cov_robust), 1e-12, None))
    t_robust = flat_init.detach().double().cpu().numpy() / std_robust

    hess = HessianResult(
        theta=flat_init.detach(),
        std=torch.tensor(std_classic),
        tstat=torch.tensor(t_classic),
        hessian=H.detach(),
        var_covar=torch.tensor(cov_classic),
        names=names,
    )
    return hess, diag, std_robust, t_robust
