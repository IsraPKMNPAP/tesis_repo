from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src.models.icl_v import DeterministicICLV


def load_run_args(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    return meta.get("args", {})


def load_split_subjects(run_dir: Path) -> List[str]:
    info = json.loads((run_dir / "split_info.json").read_text(encoding="utf-8"))
    return info.get("train_subjects", [])


def to_float_array(mat) -> np.ndarray:
    try:
        arr = mat.toarray()
    except Exception:
        arr = np.asarray(mat)
    return arr.astype(np.float32)


def prepare_preprocessor(
    df: pd.DataFrame, cols: List[str], scaler: str, cat_unique_threshold: int
) -> Tuple[np.ndarray, ColumnTransformer]:
    df_prep = df[cols].copy()
    for c in df_prep.columns:
        if df_prep[c].dtype == object:
            df_prep[c] = df_prep[c].astype("category")
        else:
            try:
                if df_prep[c].nunique(dropna=True) <= cat_unique_threshold:
                    df_prep[c] = df_prep[c].astype("category")
            except Exception:
                pass
    numeric = df_prep.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical = df_prep.select_dtypes(include=["category"]).columns.tolist()
    scaler_cls = RobustScaler if scaler == "robust" else StandardScaler
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler_cls(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical),
        ]
    )
    mat = preprocessor.fit_transform(df_prep)
    return mat, preprocessor


def filter_low_variance(mat: np.ndarray, feature_names: List[str], min_var: float) -> Tuple[np.ndarray, List[str], np.ndarray]:
    X = to_float_array(mat)
    if X.shape[1] == 0:
        return X, feature_names, np.ones(0, dtype=bool)
    var = np.var(X, axis=0)
    mask = var >= min_var
    kept = [name for name, keep in zip(feature_names, mask) if keep]
    return X[:, mask], kept, mask


def build_design_matrices(
    df: pd.DataFrame,
    obs_lt_cols: List[str],
    obs_u_cols: List[str],
    n_choices: int,
    scaler: str,
    cat_unique_threshold: int,
    min_var: float,
    obs_u_buy_only: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X_lt, preproc_lt = prepare_preprocessor(df, obs_lt_cols, scaler, cat_unique_threshold) if obs_lt_cols else (np.zeros((len(df), 0), dtype=np.float32), None)
    X_u, preproc_u = prepare_preprocessor(df, obs_u_cols, scaler, cat_unique_threshold)
    try:
        feat_names_u = list(preproc_u.get_feature_names_out(obs_u_cols))
    except Exception:
        feat_names_u = list(obs_u_cols)
    X_u, feat_names_u, mask = filter_low_variance(X_u, feat_names_u, min_var)
    if obs_u_buy_only:
        X_u2 = np.zeros((len(X_u), n_choices, X_u.shape[1]), dtype=np.float32)
        X_u2[:, 1, :] = X_u
        X_u = X_u2
    else:
        X_u = to_float_array(X_u)
        if X_u.ndim == 2:
            X_u = X_u[:, None, :].repeat(n_choices, axis=1)
    return to_float_array(X_lt), X_u, feat_names_u


def bootstrap_indices(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(len(y)), size=n, replace=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap de betas de utilidad para ICLV (reajuste solo beta).")
    parser.add_argument("--iclv-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--by-subject", action="store_true", help="Bootstrap por sujeto.")
    args = parser.parse_args()

    run_args = load_run_args(args.iclv_dir)
    data_path = args.data or Path(run_args.get("data", ""))
    if not data_path:
        raise ValueError("Se requiere --data o run_metadata.json con data.")

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()
    if "subject" not in df.columns and "id_sub" in df.columns:
        df["subject"] = df["id_sub"].astype(str)
    train_subjects = set(load_split_subjects(args.iclv_dir))
    if train_subjects:
        df = df[df["subject"].isin(train_subjects)].reset_index(drop=True)

    label_col = run_args.get("label_col", "bought").lower()
    y = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=np.int64)

    obs_lt_cols = [c.strip().lower() for c in (run_args.get("obs_lt_cols") or [])] if isinstance(run_args.get("obs_lt_cols"), list) else []
    obs_u_cols = [c.strip().lower() for c in (run_args.get("obs_u_cols") or [])] if isinstance(run_args.get("obs_u_cols"), list) else []
    if not obs_u_cols:
        obs_u_cols = json.loads((args.iclv_dir / "metrics.json").read_text(encoding="utf-8")).get("obs_u_cols", [])
    n_choices = int(run_args.get("num_choices", 2))
    scaler = run_args.get("scaler", "standard")
    cat_unique_threshold = int(run_args.get("cat_unique_threshold", 4))
    min_var = float(run_args.get("min_var", 1e-6))
    obs_u_buy_only = bool(run_args.get("obs_u_buy_only", False))
    n_latent = int(run_args.get("n_latent", 1))

    X_lt, X_u, feat_names_u = build_design_matrices(
        df,
        obs_lt_cols,
        obs_u_cols,
        n_choices,
        scaler,
        cat_unique_threshold,
        min_var,
        obs_u_buy_only,
    )

    device = torch.device(args.device)
    base_model = DeterministicICLV(
        dim_obs_lt=X_lt.shape[1],
        dim_obs_u=X_u.shape[2],
        n_latent=n_latent,
        n_indicators=0,
        n_choices=n_choices,
        delta_per_alt=True,
        beta_per_alt=bool(run_args.get("beta_per_alt", False)),
    ).to(device)
    state = torch.load(args.iclv_dir / "model.pt", map_location="cpu", weights_only=True)
    base_model.load_state_dict(state, strict=False)

    beta_param = base_model.beta if isinstance(base_model.beta, torch.nn.Parameter) else base_model.beta.weight
    beta_dim = beta_param.numel()
    beta_boot = np.zeros((args.n_bootstrap, beta_dim), dtype=np.float64)

    for b in range(args.n_bootstrap):
        if args.by_subject and "subject" in df.columns:
            rng = np.random.default_rng(args.seed + b)
            subs = rng.choice(df["subject"].unique(), size=len(df["subject"].unique()), replace=True)
            idx = df[df["subject"].isin(subs)].index.to_numpy()
        else:
            idx = bootstrap_indices(y, len(y), seed=args.seed + b)

        X_lt_b = torch.tensor(X_lt[idx], dtype=torch.float32, device=device)
        X_u_b = torch.tensor(X_u[idx], dtype=torch.float32, device=device)
        y_b = torch.tensor(y[idx], dtype=torch.long, device=device)

        model = deepcopy(base_model)
        for p in model.parameters():
            p.requires_grad = False
        beta_param_b = model.beta if isinstance(model.beta, torch.nn.Parameter) else model.beta.weight
        beta_param_b.requires_grad = True
        optimizer = torch.optim.Adam([beta_param_b], lr=args.lr)

        for _ in range(args.max_iter):
            out = model(obs_lt=X_lt_b, obs_u=X_u_b, indicators=None, choice=y_b)
            loss = -out["loglik_choice_sum"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        beta_boot[b, :] = beta_param_b.detach().cpu().numpy().reshape(-1)

    beta_mean = beta_boot.mean(axis=0)
    beta_std = beta_boot.std(axis=0, ddof=1)
    tstat = beta_mean / np.where(beta_std == 0, np.nan, beta_std)

    rows = []
    base_alt = 0
    alt_list = [a for a in range(n_choices) if a != base_alt]
    for alt_pos, alt in enumerate(alt_list):
        for j, feat in enumerate(feat_names_u):
            idx = alt_pos * len(feat_names_u) + j
            if idx >= beta_mean.shape[0]:
                continue
            rows.append(
                {
                    "alt": alt,
                    "feature": feat,
                    "coef": float(beta_mean[idx]),
                    "std": float(beta_std[idx]) if beta_std[idx] == beta_std[idx] else np.nan,
                    "tstat": float(tstat[idx]) if tstat[idx] == tstat[idx] else np.nan,
                }
            )

    out_df = pd.DataFrame(rows)
    out_path = args.iclv_dir / "utility_stats_bootstrap.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} (rows: {len(out_df)})")


if __name__ == "__main__":
    main()
