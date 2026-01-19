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
from biogeme.expressions import Beta, Variable, log

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
    print(f"[mnl] one-hot cols: {cat_cols} -> {len(dummies.columns)} dummies")
    return df, new_cols


class TorchMNL(nn.Module):
    def __init__(self, dim_obs_u: int):
        super().__init__()
        self.asc = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(dim_obs_u))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D] for alt=1, base alt=0
        u1 = self.asc + x @ self.beta
        u0 = torch.zeros_like(u1)
        logits = torch.stack([u0, u1], dim=1)
        return logits


def torch_param_stats(model: TorchMNL, X: torch.Tensor, y: torch.Tensor) -> pd.DataFrame:
    params = [p for p in model.parameters() if p.requires_grad]
    flat_init = torch.nn.utils.parameters_to_vector(params).detach()

    def _nll(flat_params: torch.Tensor) -> torch.Tensor:
        torch.nn.utils.vector_to_parameters(flat_params, params)
        logits = model(X)
        nll = F.nll_loss(F.log_softmax(logits, dim=1), y, reduction="sum")
        return nll

    H = torch.autograd.functional.hessian(_nll, flat_init)
    eye = torch.eye(H.shape[0], device=H.device, dtype=H.dtype) * 1e-6
    H_inv = torch.linalg.pinv(H + eye)
    var = torch.diag(H_inv)
    std = torch.sqrt(torch.clamp(var, min=1e-12))
    theta = flat_init
    tstat = theta / torch.clamp(std, min=1e-12)

    names = ["ASC_1"] + [f"beta_{i}" for i in range(model.beta.numel())]
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
    k = int(p1.size * 0 + 0)  # placeholder, filled by caller
    aic = float("nan")
    bic = float("nan")
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
        "aic": aic,
        "bic": bic,
        "loglik_null": loglik_null,
        "loglik_ratio": loglik_ratio,
        "pseudo_r2": pseudo_r2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparacion MNL Torch vs Biogeme (misma especificacion).")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--obs-u-cols", type=Path, required=True)
    parser.add_argument("--label-col", type=str, default="bought")
    parser.add_argument("--cat-unique-threshold", type=int, default=4)
    parser.add_argument("--standardize-numeric-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--results-dir", type=Path, default=Path("./results/mnl_compare"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.lower()
    label_col = args.label_col.lower()
    obs_u_cols = [c.strip().lower() for c in load_features_file(args.obs_u_cols)]
    obs_u_cols = [c for c in obs_u_cols if c in df.columns]
    if not obs_u_cols:
        raise ValueError("obs_u_cols quedo vacio.")

    df, obs_u_cols = expand_categoricals(
        df,
        obs_u_cols,
        prefix="u_",
        cat_unique_threshold=args.cat_unique_threshold,
        standardize_numeric=args.standardize_numeric_only,
    )
    keep_cols = [label_col] + obs_u_cols
    df = df[keep_cols].copy()
    for c in keep_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    y = df[label_col].to_numpy(dtype=int)
    X = df[obs_u_cols].to_numpy(dtype=np.float64)

    # Torch MNL (full-batch)
    device = torch.device("cpu")
    model = TorchMNL(dim_obs_u=X.shape[1]).to(device).double()
    X_t = torch.tensor(X, dtype=torch.float64, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    opt = torch.optim.LBFGS(model.parameters(), lr=args.lr, max_iter=args.epochs, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = model(X_t)
        loss = F.nll_loss(F.log_softmax(logits, dim=1), y_t, reduction="sum")
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        logits = model(X_t)
        p1_t = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    torch_metrics = compute_metrics(y, p1_t)
    k_torch = int(sum(p.numel() for p in model.parameters()))
    torch_metrics["aic"] = float(2 * k_torch - 2 * torch_metrics["log_likelihood"])
    torch_metrics["bic"] = float(np.log(max(1, len(y))) * k_torch - 2 * torch_metrics["log_likelihood"])
    torch_metrics["n_params"] = k_torch
    torch_metrics["n_obs"] = int(len(y))

    # Biogeme MNL
    database = db.Database("neuma", df)
    Choice = Variable(label_col)
    obs_u_vars = [Variable(c) for c in obs_u_cols]
    ASC1 = Beta("ASC_1", 0, None, None, 0)
    beta_u = [Beta(f"beta_{c}", 0, None, None, 0) for c in obs_u_cols]
    U1 = ASC1 + sum(b * x for b, x in zip(beta_u, obs_u_vars))
    V = {0: 0, 1: U1}
    av = {0: 1, 1: 1}
    P = models.logit(V, av, Choice)
    logprob = log(P)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    biogeme = bio.BIOGEME(database, logprob)
    biogeme.model_name = "mnl_biogeme"
    results = biogeme.estimate()
    beta_values = results.getBetaValues() if hasattr(results, "getBetaValues") else results.get_beta_values()
    sim = bio.BIOGEME(database, {"p1": models.logit(V, av, 1)})
    sim.model_name = "mnl_biogeme_sim"
    sim_res = sim.simulate(beta_values)
    p1_b = sim_res["p1"].to_numpy(dtype=float)

    bio_metrics = compute_metrics(y, p1_b)
    k_bio = int(len(beta_values))
    bio_metrics["aic"] = float(2 * k_bio - 2 * bio_metrics["log_likelihood"])
    bio_metrics["bic"] = float(np.log(max(1, len(y))) * k_bio - 2 * bio_metrics["log_likelihood"])
    bio_metrics["n_params"] = k_bio
    bio_metrics["n_obs"] = int(len(y))

    torch_params = torch_param_stats(model, X_t, y_t)
    torch_params["name"] = ["ASC_1"] + [f"beta_{c}" for c in obs_u_cols]
    torch_params.to_csv(args.results_dir / "torch_mnl_params.csv", index=False)

    out = {
        "torch_mnl": torch_metrics,
        "biogeme_mnl": bio_metrics,
        "n_obs": int(len(y)),
        "n_features": int(X.shape[1]),
        "features": obs_u_cols,
    }
    out_path = args.results_dir / "mnl_compare_metrics.json"
    pd.Series(out).to_json(out_path, indent=2, force_ascii=False)
    print(f"Saved metrics: {out_path}")


if __name__ == "__main__":
    main()
