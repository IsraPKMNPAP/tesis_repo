from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class ICLV(nn.Module):
    """
    ICLV con Estimación por Máxima Verosimilitud Simulada (SML).
    
    Cambios principales:
    1. Integración de Monte Carlo en el forward (n_draws).
    2. Cálculo de log-probabilidad sobre el promedio de probabilidades (Integral).
    3. Manejo de escala de indicadores integrado en la verosimilitud conjunta.
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
        self.n_latent = n_latent
        self.alpha = float(alpha)
        self.beta_per_alt = bool(beta_per_alt)
        self.base_alt = 0
        jm1 = self.n_choices - 1

        # Bloque estructural (Variable Latente)
        self.Gamma = nn.Linear(dim_obs_lt, n_latent)
        # Bloque de medición (Indicadores)
        self.Lambda = nn.Linear(n_latent, n_indicators) if n_indicators > 0 else None

        # Bloque de utilidad
        if self.beta_per_alt:
            self.beta = nn.Parameter(torch.zeros(jm1, dim_obs_u))
        else:
            self.beta = nn.Linear(dim_obs_u, 1, bias=False)
            
        if delta_per_alt:
            self.delta = nn.Parameter(torch.zeros(jm1, n_latent))
        else:
            self.delta = nn.Parameter(torch.zeros(n_latent))
            
        self.ASC = nn.Parameter(torch.zeros(jm1))

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
        
        # Anclaje de escala para identificación (Lambda[0,0] = 1)
        if self.Lambda is not None and self.Lambda.weight.numel() > 0:
            with torch.no_grad():
                self.Lambda.weight[0, 0] = 1.0
            self.Lambda.weight.register_hook(self._freeze_lambda_anchor_grad)

    def _freeze_lambda_anchor_grad(self, grad: torch.Tensor) -> torch.Tensor:
        if grad is None or grad.numel() == 0:
            return grad
        grad = grad.clone()
        grad[0, 0] = 0.0 # Congela el gradiente del ancla
        return grad

    def compute_utilities(self, obs_u: torch.Tensor, LT: torch.Tensor) -> torch.Tensor:
        """V_nj = beta^T OBS_U_nj + delta_j^T LT_n + ASC_j."""
        B, J, _ = obs_u.shape
        device = obs_u.device
        dtype = obs_u.dtype
        
        asc_full = torch.zeros(J, device=device, dtype=dtype)
        if self.ASC.numel() > 0:
            asc_full[1:] = self.ASC
            
        if self.beta_per_alt:
            beta_full = torch.zeros(J, obs_u.size(-1), device=device, dtype=dtype)
            beta_full[1:, :] = self.beta
            beta_term = (obs_u * beta_full.unsqueeze(0)).sum(-1)
        else:
            beta_term = self.beta(obs_u).squeeze(-1)

        if self.delta.dim() == 2:
            delta_full = torch.zeros(J, LT.size(-1), device=device, dtype=dtype)
            delta_full[1:, :] = self.delta
            delta_term = LT @ delta_full.t()
        else:
            delta_term = (LT @ self.delta).unsqueeze(1).expand_as(beta_term)
            
        return beta_term + delta_term + asc_full.unsqueeze(0)

    def forward(
        self,
        obs_lt: torch.Tensor,
        obs_u: torch.Tensor,
        indicators: torch.Tensor,
        choice: torch.Tensor,
        n_draws: int = 50 # SML: Número de extracciones para la integral
    ):
        B = obs_lt.size(0)
        device = obs_lt.device
        
        # 1. Media de la variable latente
        LT_mean = self.Gamma(obs_lt) 
        
        # 2. SML: Generar extracciones de Monte Carlo para integrar el error latente
        # LT = Gamma*X + epsilon, donde epsilon ~ N(0,1)
        eps = torch.randn(n_draws, B, self.n_latent, device=device)
        LT_samples = LT_mean.unsqueeze(0) + eps # [n_draws, B, n_latent]
        
        # Aplanar para procesar todas las extracciones eficientemente
        LT_flat = LT_samples.view(-1, self.n_latent) # [n_draws*B, n_latent]
        obs_u_rep = obs_u.repeat_interleave(n_draws, dim=0) # [n_draws*B, J, D]
        
        # 3. Parte de Elección: Promedio de Probabilidades (SML)
        V_flat = self.compute_utilities(obs_u_rep, LT_flat) # [n_draws*B, J]
        probs_flat = F.softmax(V_flat, dim=1)
        
        # Regresar a forma [n_draws, B, J] y promediar probabilidades
        probs_samples = probs_flat.view(n_draws, B, -1)
        mean_probs = probs_samples.mean(dim=0) # E[P(j)]
        logp = torch.log(mean_probs + 1e-9)
        ll_choice = logp.gather(1, choice.view(-1, 1)).sum()
        
        # 4. Parte de Medición: Integración de la densidad de indicadores
        if self.Lambda is not None and self.n_indicators > 0:
            I_hat_flat = self.Lambda(LT_flat)
            I_hat_samples = I_hat_flat.view(n_draws, B, -1)
            
            # Densidad de una Normal (asumiendo sigma=1)
            # log f(I) = -0.5 * (I - I_hat)^2
            dist_sq = -0.5 * torch.pow(indicators.unsqueeze(0) - I_hat_samples, 2).sum(dim=-1)
            # log(mean(exp(densidades))) -> logsumexp - log(n_draws)
            ll_meas = torch.logsumexp(dist_sq, dim=0).sum() - B * torch.log(torch.tensor(float(n_draws), device=device))
        else:
            ll_meas = torch.tensor(0.0, device=device)

        total_loglik = ll_choice + self.alpha * ll_meas
        
        return {
            "loss": -total_loglik,
            "log_likelihood": total_loglik,
            "loglik_choice_sum": ll_choice,
            "loglik_meas_sum": ll_meas,
            "probs": mean_probs
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
    batch: dict,
    n_draws: int = 100, # Aumentar draws para estabilidad del Hessiano
    ridge: float = 1e-6,
) -> HessianResult:
    """Calcula el Hessiano completo sobre la Log-Likelihood Simulada."""
    named_params = trainable_named_params(model)
    params = [p for _, p in named_params]
    names = param_names(model)
    flat_init = parameters_to_vector(params).detach()

    # Fijamos semilla para que el Hessiano numérico sea estable entre perturbaciones
    torch.manual_seed(42) 

    def _wrapped_nll(flat_params: torch.Tensor) -> torch.Tensor:
        vector_to_parameters(flat_params, params)
        out = model(
            obs_lt=batch["obs_lt"],
            obs_u=batch["obs_u"],
            indicators=batch.get("indicators"),
            choice=batch["choice"],
            n_draws=n_draws
        )
        return -out["log_likelihood"]

    H = torch.autograd.functional.hessian(_wrapped_nll, flat_init)
    vector_to_parameters(flat_init, params)

    eye = torch.eye(H.shape[0], device=H.device) * ridge
    H_inv = torch.linalg.pinv(H + eye)

    var = torch.diag(H_inv)
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    tstat = flat_init / std

    return HessianResult(
        theta=flat_init.detach(),
        std=std.detach(),
        tstat=tstat.detach(),
        hessian=H.detach(),
        var_covar=H_inv.detach(),
        names=names,
    )