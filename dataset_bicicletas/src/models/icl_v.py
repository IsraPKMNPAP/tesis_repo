from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .mm_vae import TabularEncoder, VideoEncoderWrapper
from .audio_encoders import SimpleAudioEncoder


class DeterministicICLV(nn.Module):
    """Deterministic amortized ICLV (sin integracion Monte Carlo).

    Estructura:
      - Latentes (LT): LT = Gamma(OBS_LT)
      - Indicadores (OBS_I): I_hat = Lambda(LT)
      - Utilidad por alternativa j: U_j = ASC_j + beta^T OBS_U_j + delta_j^T LT
      - P(y=j) = softmax(U)_j
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
    ):
        super().__init__()
        self.n_choices = int(n_choices)
        self.n_indicators = int(n_indicators)
        self.alpha = float(alpha)

        # Bloque estructural (Gamma) y de medicion (Lambda)
        self.Gamma = nn.Linear(dim_obs_lt, n_latent)
        self.Lambda = nn.Linear(n_latent, n_indicators) if n_indicators > 0 else None

        # Bloque de utilidad
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
        nn.init.xavier_uniform_(self.beta.weight)
        nn.init.zeros_(self.ASC)
        nn.init.zeros_(self.delta)

    def compute_utilities(self, obs_u: torch.Tensor, LT: torch.Tensor) -> torch.Tensor:
        """Computa V_nj = beta^T OBS_U_nj + delta_j^T LT_n + ASC_j."""
        if obs_u.dim() != 3:
            raise ValueError(f"Se espera obs_u con shape [B, J, dim_obs_u]; se recibio {obs_u.shape}")
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
    """Calcula Hessiano numerico, errores estandar y t-stats sobre la loss provista."""
    named_params = trainable_named_params(model)
    params: Sequence[nn.Parameter] = [p for _, p in named_params]
    names = param_names(model)

    flat_init = parameters_to_vector(params).detach()

    def _wrapped_loss(flat_params: torch.Tensor) -> torch.Tensor:
        vector_to_parameters(flat_params, params)
        return loss_closure()

    H = torch.autograd.functional.hessian(_wrapped_loss, flat_init)
    # Restaurar parametros originales
    vector_to_parameters(flat_init, params)

    # Estabilizar inversion
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


class MultimodalICLVDeterministic(nn.Module):
    """ICLV con encoder multimodal determinista (tab OBS_LT + video + audio -> z)."""

    def __init__(
        self,
        tab_in_dim: int,
        dim_obs_u: int,
        n_indicators: int,
        n_choices: int,
        tab_emb_dim: int = 128,
        shared_dim: int = 64,
        vid_backbone: nn.Module | None = None,
        audio_encoder: nn.Module | None = None,
        alpha: float = 1.0,
        delta_per_alt: bool = True,
        fuse_dropout: float = 0.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.n_indicators = int(n_indicators)
        self.n_choices = int(n_choices)

        self.tab_enc = TabularEncoder(tab_in_dim, tab_emb_dim, dropout=fuse_dropout)
        self.vid_enc = VideoEncoderWrapper(backbone_model=vid_backbone)
        self.audio_enc = audio_encoder if audio_encoder is not None else SimpleAudioEncoder(emb_dim=tab_emb_dim)

        vid_emb_dim = self.vid_enc.output_dim()
        aud_emb_dim = tab_emb_dim  # SimpleAudioEncoder outputs emb_dim
        fuse_in = tab_emb_dim + vid_emb_dim + aud_emb_dim
        fuse_hidden = max(shared_dim * 2, fuse_in // 2 + 1)
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, fuse_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=fuse_dropout) if fuse_dropout and fuse_dropout > 0 else nn.Identity(),
            nn.Linear(fuse_hidden, shared_dim),
        )

        # Measurement block
        self.Lambda = nn.Linear(shared_dim, n_indicators) if n_indicators > 0 else None

        # Utility block
        self.beta = nn.Linear(dim_obs_u, 1, bias=False)
        if delta_per_alt:
            self.delta = nn.Parameter(torch.zeros(n_choices, shared_dim))
        else:
            self.delta = nn.Parameter(torch.zeros(shared_dim))
        self.ASC = nn.Parameter(torch.zeros(n_choices))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.tab_enc.net[-1].weight)
        if hasattr(self.tab_enc.net[-1], "bias") and self.tab_enc.net[-1].bias is not None:
            nn.init.zeros_(self.tab_enc.net[-1].bias)
        if self.Lambda is not None:
            nn.init.xavier_uniform_(self.Lambda.weight)
            if self.Lambda.bias is not None:
                nn.init.zeros_(self.Lambda.bias)
        nn.init.xavier_uniform_(self.beta.weight)
        nn.init.zeros_(self.ASC)
        nn.init.zeros_(self.delta)

    def compute_utilities(self, obs_u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if obs_u.dim() != 3:
            raise ValueError(f"Se espera obs_u con shape [B, J, dim_obs_u]; se recibio {obs_u.shape}")
        beta_term = self.beta(obs_u).squeeze(-1)  # [B, J]
        if self.delta.dim() == 2:
            delta_term = z @ self.delta.t()  # [B, J]
        else:
            delta_term = (z @ self.delta).unsqueeze(1).expand_as(beta_term)
        asc_term = self.ASC.unsqueeze(0)
        return beta_term + delta_term + asc_term

    def forward(
        self,
        x_tab_lt: torch.Tensor,
        x_vid: torch.Tensor,
        x_aud: torch.Tensor | None,
        obs_u: torch.Tensor,
        indicators: torch.Tensor,
        choice: torch.Tensor,
    ):
        z_tab = self.tab_enc(x_tab_lt)
        z_vid = self.vid_enc(x_vid)
        z_aud = self.audio_enc(x_aud) if x_aud is not None else torch.zeros_like(z_tab)
        z = torch.cat([z_tab, z_vid, z_aud], dim=1)
        z = self.fuse(z)

        I_hat = self.Lambda(z) if (self.Lambda is not None and self.n_indicators > 0) else None
        V = self.compute_utilities(obs_u, z)
        logp = F.log_softmax(V, dim=1)

        loss_choice = F.nll_loss(logp, choice, reduction="mean")
        if I_hat is None:
            loss_meas = torch.tensor(0.0, device=z.device, dtype=z.dtype)
        else:
            loss_meas = F.mse_loss(I_hat, indicators, reduction="mean")
        loss = loss_choice + self.alpha * loss_meas
        ll = logp.gather(1, choice.view(-1, 1)).sum()

        return {
            "loss": loss,
            "loss_choice": loss_choice,
            "loss_meas": loss_meas,
            "logp": logp,
            "z": z,
            "I_hat": I_hat,
            "log_likelihood": ll,
        }
