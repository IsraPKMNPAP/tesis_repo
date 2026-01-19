from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, Draws, MonteCarlo, log, exp

from utils.features import load_features_file


def expand_categoricals(
    df: pd.DataFrame,
    cols: list[str],
    prefix: str,
    cat_unique_threshold: int,
    standardize_numeric: bool,
) -> tuple[pd.DataFrame, list[str]]:
    if not cols:
        return df, cols
    cat_cols = []
    num_cols = []
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique(dropna=True) <= cat_unique_threshold:
            cat_cols.append(c)
        else:
            num_cols.append(c)
    if standardize_numeric and num_cols:
        for c in num_cols:
            col = pd.to_numeric(df[c], errors="coerce")
            mu = col.mean()
            sd = col.std()
            if sd and not np.isnan(sd):
                df[c] = (col - mu) / sd
            else:
                df[c] = col - mu
    if not cat_cols:
        return df, cols
    dummies = pd.get_dummies(df[cat_cols].astype(str), prefix=[f"{prefix}{c}" for c in cat_cols], drop_first=True)
    df = df.drop(columns=cat_cols).join(dummies)
    new_cols = num_cols + dummies.columns.tolist()
    print(f"[iclv] one-hot cols: {cat_cols} -> {len(dummies.columns)} dummies")
    return df, new_cols


class TorchICLV(nn.Module):
    def __init__(self, dim_obs_lt: int, dim_obs_u: int, dim_obs_i: int, n_latent: int, use_latent: bool):
        super().__init__()
        self.n_latent = n_latent
        self.use_latent = bool(use_latent)
        self.asc = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(dim_obs_u))
        self.gamma = nn.Parameter(torch.zeros(n_latent, dim_obs_lt))
        self.delta = nn.Parameter(torch.zeros(n_latent))
        self.alpha = nn.Parameter(torch.zeros(dim_obs_i)) if dim_obs_i > 0 else None
        self.lambda_ = nn.Parameter(torch.zeros(dim_obs_i, n_latent)) if dim_obs_i > 0 else None
        if self.lambda_ is not None and self.lambda_.numel() > 0:
            with torch.no_grad():
                self.lambda_[0, 0] = 1.0
                if n_latent >= 2 and self.lambda_.shape[0] > 1:
                    self.lambda_[1, 1] = 1.0
            self.lambda_.register_hook(self._freeze_lambda_grad)

    def _freeze_lambda_grad(self, grad: torch.Tensor) -> torch.Tensor:
        grad = grad.clone()
        grad[0, 0] = 0.0
        if self.n_latent >= 2 and grad.shape[0] > 1:
            grad[1, 1] = 0.0
        return grad

    def forward(self, X_lt, X_u, X_i, y, n_draws: int, with_meas: bool):
        B = X_lt.shape[0]
        if self.use_latent:
            LT_mean = X_lt @ self.gamma.t()  # [B, L]
            eps = torch.randn(n_draws, B, self.n_latent, device=X_lt.device)
            LT = LT_mean.unsqueeze(0) + eps  # [S, B, L]
            LT_flat = LT.view(-1, self.n_latent)
            X_u_rep = X_u.repeat_interleave(n_draws, dim=0)
            u1 = self.asc + X_u_rep @ self.beta + LT_flat @ self.delta
        else:
            n_draws = 1
            LT = torch.zeros(1, B, self.n_latent, device=X_lt.device)
            LT_flat = LT.view(-1, self.n_latent)
            X_u_rep = X_u
            u1 = self.asc + X_u_rep @ self.beta
        u0 = torch.zeros_like(u1)
        logits = torch.stack([u0, u1], dim=1)
        probs = F.softmax(logits, dim=1)
        probs = probs.view(n_draws, B, 2).mean(dim=0)
        logp = torch.log(probs + 1e-9)
        ll_choice = logp.gather(1, y.view(-1, 1)).sum()

        ll_meas = torch.tensor(0.0, device=X_lt.device)
        if with_meas and self.use_latent and X_i is not None and X_i.shape[1] > 0:
            I_hat = self.alpha + (LT @ self.lambda_.t())
            dist = -0.5 * torch.pow(X_i.unsqueeze(0) - I_hat, 2).sum(dim=-1)
            ll_meas = (torch.logsumexp(dist, dim=0) - torch.log(torch.tensor(float(n_draws)))).sum()

        ll = ll_choice + ll_meas
        loss = -ll
        return loss, ll, ll_choice, ll_meas, probs


def compute_metrics(y: np.ndarray, p1: np.ndarray) -> dict:
    p1 = np.clip(p1, 1e-9, 1 - 1e-9)
    y_hat = (p1 >= 0.5).astype(int)
    acc = float(accuracy_score(y, y_hat))
    f1_macro = float(f1_score(y, y_hat, average="macro"))
    f1_pos = float(f1_score(y, y_hat, pos_label=1))
    f1_neg = float(f1_score(y, y_hat, pos_label=0))
    auc = float(roc_auc_score(y, p1)) if len(np.unique(y)) > 1 else float("nan")
    loglik = float(np.sum(y * np.log(p1) + (1 - y) * np.log(1 - p1)))
    nll = float(-loglik)
    mean_nll = float(-loglik / max(1, len(y)))
    p_null = y.mean()
    p_null = float(min(max(p_null, 1e-9), 1 - 1e-9))
    loglik_null = float(np.sum(y * np.log(p_null) + (1 - y) * np.log(1 - p_null)))
    loglik_ratio = float(2 * (loglik - loglik_null))
    pseudo_r2 = float(1 - (loglik / loglik_null)) if loglik_null != 0 else float("nan")
    return {
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_pos": f1_pos,
        "f1_neg": f1_neg,
        "auc": auc,
        "nll": nll,
        "mean_nll": mean_nll,
        "log_likelihood": loglik,
        "loglik_null": loglik_null,
        "loglik_ratio": loglik_ratio,
        "pseudo_r2": pseudo_r2,
    }


def torch_param_stats(model: TorchICLV, X_lt, X_u, X_i, y, n_draws: int, with_meas: bool) -> pd.DataFrame:
    params = [p for p in model.parameters() if p.requires_grad]
    flat_init = torch.nn.utils.parameters_to_vector(params).detach()

    def _nll(flat_params: torch.Tensor) -> torch.Tensor:
        torch.nn.utils.vector_to_parameters(flat_params, params)
        loss, _, _ = model(X_lt, X_u, X_i, y, n_draws=n_draws, with_meas=with_meas)
        return loss

    H = torch.autograd.functional.hessian(_nll, flat_init)
    eye = torch.eye(H.shape[0], device=H.device, dtype=H.dtype) * 1e-6
    H_inv = torch.linalg.pinv(H + eye)
    var = torch.diag(H_inv)
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    theta = flat_init
    tstat = theta / torch.clamp(std, min=1e-12)

    names = ["ASC_1"]
    names += [f"beta_{i}" for i in range(model.beta.numel())]
    names += [f"gamma_{i}" for i in range(model.gamma.numel())]
    names += [f"delta_lv_{i}" for i in range(model.delta.numel())]
    if model.alpha is not None:
        names += [f"alpha_i{i}" for i in range(model.alpha.numel())]
    if model.lambda_ is not None:
        names += [f"lambda_{i}" for i in range(model.lambda_.numel())]

    rows = []
    for name, b, s, t in zip(names, theta.detach().cpu().numpy(), std.detach().cpu().numpy(), tstat.detach().cpu().numpy()):
        if abs(t) >= 2.58:
            stars = "***"
        elif abs(t) >= 1.96:
            stars = "**"
        elif abs(t) >= 1.64:
            stars = "*"
        else:
            stars = ""
        rows.append({"name": name, "beta": float(b), "std": float(s), "tstat": float(t), "stars": stars})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="ICLV Torch vs Biogeme (SML + medicion).")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--obs-lt-cols", type=Path, required=True)
    parser.add_argument("--obs-u-cols", type=Path, required=True)
    parser.add_argument("--obs-i-cols", type=Path, required=True)
    parser.add_argument("--label-col", type=str, default="bought")
    parser.add_argument("--n-draws", type=int, default=200)
    parser.add_argument("--n-latent", type=int, default=1)
    parser.add_argument("--cat-unique-threshold", type=int, default=4)
    parser.add_argument("--standardize-numeric-only", action="store_true")
    parser.add_argument("--with-measurement", action="store_true")
    parser.add_argument("--no-latent", action="store_true", help="Desactiva bloque latente (MNL puro).")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--results-dir", type=Path, default=Path("./results/icl_v_compare"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.lower()
    label_col = args.label_col.lower()

    obs_lt_cols = [c.strip().lower() for c in load_features_file(args.obs_lt_cols)]
    obs_u_cols = [c.strip().lower() for c in load_features_file(args.obs_u_cols)]
    obs_i_cols = [c.strip().lower() for c in load_features_file(args.obs_i_cols)]
    obs_lt_cols = [c for c in obs_lt_cols if c in df.columns]
    obs_u_cols = [c for c in obs_u_cols if c in df.columns]
    obs_i_cols = [c for c in obs_i_cols if c in df.columns]

    df, obs_lt_cols = expand_categoricals(df, obs_lt_cols, "lt_", args.cat_unique_threshold, args.standardize_numeric_only)
    df, obs_u_cols = expand_categoricals(df, obs_u_cols, "u_", args.cat_unique_threshold, args.standardize_numeric_only)
    df, obs_i_cols = expand_categoricals(df, obs_i_cols, "i_", args.cat_unique_threshold, args.standardize_numeric_only)

    keep_cols = [label_col] + obs_lt_cols + obs_u_cols + obs_i_cols
    df = df[keep_cols].copy()
    for c in keep_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    y = df[label_col].to_numpy(dtype=int)
    X_lt = df[obs_lt_cols].to_numpy(dtype=np.float64)
    X_u = df[obs_u_cols].to_numpy(dtype=np.float64)
    X_i = df[obs_i_cols].to_numpy(dtype=np.float64) if obs_i_cols else np.zeros((len(df), 0), dtype=np.float64)

    device = torch.device("cpu")
    use_latent = not args.no_latent
    model = TorchICLV(X_lt.shape[1], X_u.shape[1], X_i.shape[1], args.n_latent, use_latent=use_latent).to(device).double()
    X_lt_t = torch.tensor(X_lt, dtype=torch.float64, device=device)
    X_u_t = torch.tensor(X_u, dtype=torch.float64, device=device)
    X_i_t = torch.tensor(X_i, dtype=torch.float64, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    opt = torch.optim.LBFGS(model.parameters(), lr=args.lr, max_iter=args.epochs, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss, _, _, _, _ = model(X_lt_t, X_u_t, X_i_t, y_t, n_draws=args.n_draws, with_meas=args.with_measurement)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        _, ll, ll_choice, ll_meas, p1 = model(
            X_lt_t, X_u_t, X_i_t, y_t, n_draws=args.n_draws, with_meas=args.with_measurement
        )
        p1 = p1[:, 1].cpu().numpy()

    torch_metrics = compute_metrics(y, p1)
    k_torch = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    torch_metrics["aic"] = float(2 * k_torch - 2 * torch_metrics["log_likelihood"])
    torch_metrics["bic"] = float(np.log(max(1, len(y))) * k_torch - 2 * torch_metrics["log_likelihood"])
    torch_metrics["n_params"] = k_torch
    torch_metrics["n_obs"] = int(len(y))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    torch_param_stats(model, X_lt_t, X_u_t, X_i_t, y_t, args.n_draws, args.with_measurement).to_csv(
        args.results_dir / "torch_iclv_params.csv", index=False
    )
    print(
        f"[torch] ll_total={float(ll):.4f} ll_choice={float(ll_choice):.4f} ll_meas={float(ll_meas):.4f} "
        f"beta_norm={float(model.beta.norm().item()):.4f} gamma_norm={float(model.gamma.norm().item()):.4f} "
        f"lambda_norm={(float(model.lambda_.norm().item()) if model.lambda_ is not None else 0.0):.4f}"
    )

    # Biogeme (matching specification)
    database = db.Database("neuma", df)
    Choice = Variable(label_col)
    obs_lt_vars = [Variable(c) for c in obs_lt_cols]
    obs_u_vars = [Variable(c) for c in obs_u_cols]
    obs_i_vars = [Variable(c) for c in obs_i_cols]

    if use_latent:
        gamma_betas = [
            [Beta(f"gamma_{k}_{c}", 0, None, None, 0) for c in obs_lt_cols]
            for k in range(args.n_latent)
        ]
        omegas = [Draws(f"omega_{k}", "NORMAL") for k in range(args.n_latent)]
        LVs = [
            sum(b * x for b, x in zip(gamma_betas[k], obs_lt_vars)) + omegas[k]
            for k in range(args.n_latent)
        ]
    else:
        LVs = [0.0 for _ in range(args.n_latent)]

    ASC1 = Beta("ASC_1", 0, None, None, 0)
    beta_u = [Beta(f"beta_{c}", 0, None, None, 0) for c in obs_u_cols]
    delta = [Beta(f"delta_lv_{k}", 0, None, None, 0) for k in range(args.n_latent)] if use_latent else []
    U1 = ASC1 + sum(b * x for b, x in zip(beta_u, obs_u_vars))
    if use_latent:
        U1 = U1 + sum(d * lv for d, lv in zip(delta, LVs))
    V = {0: 0, 1: U1}
    av = {0: 1, 1: 1}
    P = models.logit(V, av, Choice)

    meas_loglik = 0
    if args.with_measurement and use_latent and obs_i_vars:
        for idx, y_var in enumerate(obs_i_vars):
            alpha = Beta(f"alpha_i{idx}", 0, None, None, 0)
            lam = []
            for k in range(args.n_latent):
                fixed = 0
                init = 0
                if args.n_latent == 1 and idx == 0 and k == 0:
                    fixed = 1
                    init = 1
                elif args.n_latent >= 2:
                    if idx == 0 and k == 0:
                        fixed = 1
                        init = 1
                    elif idx == 1 and k == 1:
                        fixed = 1
                        init = 1
                lam.append(Beta(f"lambda_i{idx}_lv{k}", init, None, None, fixed))
            sigma = 1.0
            mu = alpha + sum(lam_k * lv for lam_k, lv in zip(lam, LVs))
            z = (y_var - mu) / sigma
            log_pdf = -0.5 * (np.log(2 * np.pi) + 2 * log(sigma) + z * z)
            meas_loglik += log_pdf

    integrand = P * exp(meas_loglik) if args.with_measurement else P
    logprob = log(MonteCarlo(integrand)) if args.with_measurement else log(P)

    biogeme = bio.BIOGEME(database, logprob, number_of_draws=args.n_draws)
    biogeme.model_name = "icl_v_biogeme_compare"
    results = biogeme.estimate()
    beta_values = results.getBetaValues() if hasattr(results, "getBetaValues") else results.get_beta_values()
    sim = bio.BIOGEME(database, {"p1": MonteCarlo(models.logit(V, av, 1))})
    sim.model_name = "icl_v_biogeme_compare_sim"
    sim_res = sim.simulate(beta_values)
    p1_b = sim_res["p1"].to_numpy(dtype=float)

    bio_metrics = compute_metrics(y, p1_b)
    k_bio = int(len(beta_values))
    bio_metrics["aic"] = float(2 * k_bio - 2 * bio_metrics["log_likelihood"])
    bio_metrics["bic"] = float(np.log(max(1, len(y))) * k_bio - 2 * bio_metrics["log_likelihood"])
    bio_metrics["n_params"] = k_bio
    bio_metrics["n_obs"] = int(len(y))

    out = {
        "torch_iclv": torch_metrics,
        "biogeme_iclv": bio_metrics,
        "n_obs": int(len(y)),
        "n_features_u": int(X_u.shape[1]),
        "n_features_lt": int(X_lt.shape[1]),
        "n_features_i": int(X_i.shape[1]),
        "features_u": obs_u_cols,
        "features_lt": obs_lt_cols,
        "features_i": obs_i_cols,
    }
    out_path = args.results_dir / "icl_v_compare_metrics.json"
    pd.Series(out).to_json(out_path, indent=2, force_ascii=False)
    print(f"Saved metrics: {out_path}")


if __name__ == "__main__":
    main()
