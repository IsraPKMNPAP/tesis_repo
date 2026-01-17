import torch
import torch.nn as nn
import torch.nn.functional as F

class ICLV(nn.Module):
    def __init__(self, dim_obs_lt, dim_obs_u, n_latent, n_indicators, n_choices, 
                 alpha=1.0, delta_per_alt=True, beta_per_alt=False):
        super().__init__()
        self.n_choices = n_choices
        self.n_indicators = n_indicators
        self.n_latent = n_latent
        self.alpha = alpha
        self.beta_per_alt = beta_per_alt
        jm1 = n_choices - 1

        self.Gamma = nn.Linear(dim_obs_lt, n_latent)
        self.Lambda = nn.Linear(n_latent, n_indicators) if n_indicators > 0 else None

        if self.beta_per_alt:
            self.beta = nn.Parameter(torch.zeros(jm1, dim_obs_u))
        else:
            self.beta = nn.Linear(dim_obs_u, 1, bias=False)
            
        self.delta = nn.Parameter(torch.zeros(jm1 if delta_per_alt else 1, n_latent))
        self.ASC = nn.Parameter(torch.zeros(jm1))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.Gamma.weight)
        if self.Lambda: nn.init.xavier_uniform_(self.Lambda.weight)
        if isinstance(self.beta, nn.Linear): nn.init.xavier_uniform_(self.beta.weight)
        if self.Lambda is not None:
            with torch.no_grad(): self.Lambda.weight[0, 0] = 1.0

    def forward(self, obs_lt, obs_u, indicators, choice, n_draws=50):
        B = obs_lt.size(0)
        LT_mean = self.Gamma(obs_lt) # [B, n_latent]
        
        # SML: Monte Carlo integration
        eps = torch.randn(n_draws, B, self.n_latent, device=obs_lt.device)
        LT_samples = LT_mean.unsqueeze(0) + eps # [S, B, L]
        
        # Choice Part
        LT_flat = LT_samples.view(-1, self.n_latent)
        # obs_u: [B, J, D] -> repeat for draws
        obs_u_rep = obs_u.repeat_interleave(n_draws, dim=0)
        
        # Calcular Utilidades
        V = self.compute_utilities_internal(obs_u_rep, LT_flat)
        probs_flat = F.softmax(V, dim=1)
        
        # Average probabilities across draws: SML core
        probs_samples = probs_flat.view(n_draws, B, self.n_choices)
        mean_probs = probs_samples.mean(dim=0) # [B, J]
        
        logp = torch.log(mean_probs + 1e-9)
        ll_choice = logp.gather(1, choice.view(-1, 1)).sum()

        # Measurement Part
        if self.Lambda is not None and indicators is not None:
            I_hat = self.Lambda(LT_flat).view(n_draws, B, -1)
            # Log-densidad de normal (sigma=1)
            dist = -0.5 * torch.pow(indicators.unsqueeze(0) - I_hat, 2).sum(dim=-1)
            ll_meas = (torch.logsumexp(dist, dim=0) - torch.log(torch.tensor(float(n_draws)))).sum()
        else:
            ll_meas = torch.tensor(0.0, device=obs_lt.device)

        return {"loss": -(ll_choice + self.alpha * ll_meas), "log_likelihood": ll_choice + ll_meas, 
                "loglik_choice_sum": ll_choice, "probs": mean_probs}

    def compute_utilities_internal(self, obs_u, LT):
        B_total, J, D = obs_u.shape
        asc_full = torch.cat([torch.zeros(1, device=obs_u.device), self.ASC])
        
        if self.beta_per_alt:
            beta_full = torch.cat([torch.zeros(1, D, device=obs_u.device), self.beta])
            beta_term = (obs_u * beta_full).sum(-1)
        else:
            beta_term = self.beta(obs_u).squeeze(-1)

        if self.delta.shape[0] > 1:
            delta_full = torch.cat([torch.zeros(1, self.n_latent, device=obs_u.device), self.delta])
            delta_term = torch.einsum('bl,jl->bj', LT, delta_full)
        else:
            delta_term = (LT @ self.delta.t())
            
        return beta_term + delta_term + asc_full