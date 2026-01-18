from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from src.models.multimodal_icl_v import MultimodalICLVDeterministic
from utils.features import load_features_file
from utils.run_utils import next_run_dir
from utils.splits import split_by_subject_train_val_test


def preprocess_block(
    train_df: pd.DataFrame,
    full_df: pd.DataFrame,
    cols: List[str],
    prefix: str,
    cat_unique_threshold: int,
    force_numeric: List[str] | None = None,
) -> tuple[np.ndarray, List[str]]:
    import pandas.api.types as ptypes

    num_cols = []
    cat_cols = []
    force_numeric = set(c.lower() for c in (force_numeric or []))
    for c in cols:
        if c in force_numeric:
            num_cols.append(c)
        elif not ptypes.is_numeric_dtype(train_df[c]) or train_df[c].nunique(dropna=True) <= cat_unique_threshold:
            cat_cols.append(c)
        else:
            num_cols.append(c)

    parts = []
    names = []
    if num_cols:
        means = train_df[num_cols].mean()
        stds = train_df[num_cols].std().replace(0, 1)
        num_full = full_df[num_cols].fillna(means)
        num_full = (num_full - means) / stds
        num_full.columns = [f"{prefix}{c}" for c in num_cols]
        parts.append(num_full)
        names.extend(num_full.columns.tolist())
    if cat_cols:
        cat_full = pd.get_dummies(
            full_df[cat_cols].astype(str),
            prefix=[f"{prefix}{c}" for c in cat_cols],
            drop_first=True,
        )
        # align with train categories
        cat_train = pd.get_dummies(
            train_df[cat_cols].astype(str),
            prefix=[f"{prefix}{c}" for c in cat_cols],
            drop_first=True,
        )
        cat_full = cat_full.reindex(columns=cat_train.columns, fill_value=0)
        parts.append(cat_full)
        names.extend(cat_full.columns.tolist())

    if not parts:
        return np.zeros((len(full_df), 0), dtype=np.float32), []
    X = pd.concat(parts, axis=1)
    return X.to_numpy(dtype=np.float32), names


def summarize_params(samples: np.ndarray, names: List[str]) -> pd.DataFrame:
    mean = samples.mean(axis=0)
    median = np.median(samples, axis=0)
    mean_abs = np.mean(np.abs(samples), axis=0)
    median_abs = np.median(np.abs(samples), axis=0)
    p25 = np.percentile(samples, 25, axis=0)
    p75 = np.percentile(samples, 75, axis=0)
    sign = np.sign(median)
    sign_pct = np.mean(np.sign(samples) == np.sign(median), axis=0)
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "name": name,
                "mean": float(mean[i]),
                "median": float(median[i]),
                "mean_abs": float(mean_abs[i]),
                "median_abs": float(median_abs[i]),
                "p25": float(p25[i]),
                "p75": float(p75[i]),
                "sign": float(sign[i]),
                "sign_pct": float(sign_pct[i]),
            }
        )
    return pd.DataFrame(rows)


def expand_param_names(model: torch.nn.Module) -> tuple[List[torch.nn.Parameter], List[str]]:
    params = []
    names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        if param.numel() == 1:
            names.append(name)
        else:
            names.extend([f"{name}[{i}]" for i in range(param.numel())])
    return params, names


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap multimodal: resumen de coeficientes.")
    parser.add_argument("--iclv-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_args = json.loads((args.iclv_dir / "run_metadata.json").read_text(encoding="utf-8")).get("args", {})
    data_path = args.data or Path(run_args.get("data", ""))
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()
    label = run_args.get("label_col", "bought").lower()
    img_col = run_args.get("img_emb_col", "embedding_path").lower()
    eeg_col = run_args.get("eeg_emb_col", "eeg_emb_path").lower()

    if "subject" not in df.columns and "id_sub" in df.columns:
        df["subject"] = df["id_sub"].astype(str)

    # normalize categories
    if "supermarketvisitduration" in df.columns:
        s = (
            df["supermarketvisitduration"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace("–", "-", regex=False)
        )
        mapping = {
            "<15 minutes": 10,
            "< 15 minutes": 10,
            "30-60 minutes": 45,
            ">60 minutes": 70,
            "nan": np.nan,
            "none": np.nan,
            "": np.nan,
        }
        mapped = s.map(mapping)
        numeric_fallback = pd.to_numeric(df["supermarketvisitduration"], errors="coerce")
        df["supermarketvisitduration"] = mapped.combine_first(numeric_fallback)
        if df["supermarketvisitduration"].isna().any():
            df["supermarketvisitduration"] = df["supermarketvisitduration"].fillna(
                df["supermarketvisitduration"].median()
            )
    if "offer" in df.columns:
        offer = df["offer"].astype(str).str.strip().str.lower()
        df["offer"] = np.where(offer.isin(["no", "nan", "none", "0", "0.0", ""]), "no", "yes")

    obs_lt_cols = [c.strip().lower() for c in load_features_file(run_args.get("obs_lt_cols", ""))]
    obs_u_cols = [c.strip().lower() for c in load_features_file(run_args.get("obs_u_cols", ""))]
    obs_lt_cols = [c for c in obs_lt_cols if c in df.columns]
    obs_u_cols = [c for c in obs_u_cols if c in df.columns]

    train_df, _, _, _ = split_by_subject_train_val_test(
        df, subject_col="subject", val_frac=float(run_args.get("val_frac", 0.2)), test_frac=float(run_args.get("test_frac", 0.2)), seed=int(run_args.get("seed", 42))
    )

    cat_unique_threshold = int(run_args.get("cat_unique_threshold", 4))
    X_lt, lt_names = preprocess_block(train_df, df, obs_lt_cols, "lt_", cat_unique_threshold)
    X_u, u_names = preprocess_block(
        train_df,
        df,
        obs_u_cols,
        "u_",
        cat_unique_threshold,
        force_numeric=["supermarketvisitduration"],
    )

    # preload embeddings
    img = np.stack([np.load(p).astype(np.float32).flatten() for p in df[img_col].tolist()])
    eeg = np.stack([np.load(p).astype(np.float32).flatten() for p in df[eeg_col].tolist()])
    y = df[label].astype(int).to_numpy()

    device = torch.device(args.device)
    n_choices = int(run_args.get("num_choices", 2))
    obs_u_buy_only = bool(run_args.get("obs_u_buy_only", False))
    model_params = {
        "dim_obs_lt": X_lt.shape[1],
        "dim_obs_u": X_u.shape[1],
        "dim_img_emb": img.shape[1],
        "dim_eeg_emb": eeg.shape[1],
        "n_latent": int(run_args.get("n_latent", 1)),
        "n_choices": n_choices,
        "img_proj_dim": int(run_args.get("img_proj_dim", 32)),
        "beta_per_alt": bool(run_args.get("beta_per_alt", False)),
    }

    base_model = MultimodalICLVDeterministic(**model_params).to(device)
    base_params, base_names = expand_param_names(base_model)

    params = []
    rng = np.random.default_rng(args.seed)
    for b in range(args.n_bootstrap):
        idx = rng.choice(np.arange(len(df)), size=len(df), replace=True)
        obs_lt_b = torch.tensor(X_lt[idx], dtype=torch.float32, device=device)
        obs_u_b = torch.tensor(X_u[idx], dtype=torch.float32, device=device)
        if obs_u_buy_only:
            obs_u_full = torch.zeros((len(idx), n_choices, obs_u_b.shape[1]), device=device, dtype=obs_u_b.dtype)
            obs_u_full[:, 1, :] = obs_u_b
            obs_u_b = obs_u_full
        else:
            obs_u_b = obs_u_b.unsqueeze(1).repeat(1, n_choices, 1)
        eeg_b = torch.tensor(eeg[idx], dtype=torch.float32, device=device)
        img_b = torch.tensor(img[idx], dtype=torch.float32, device=device)
        y_b = torch.tensor(y[idx], dtype=torch.long, device=device)

        model = MultimodalICLVDeterministic(**model_params).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for _ in range(args.max_iter):
            out = model(obs_lt_b, obs_u_b, eeg_b, img_b, y_b)
            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()

        params_list, _ = expand_param_names(model)
        vec = torch.nn.utils.parameters_to_vector(params_list).detach().cpu().numpy()
        params.append(vec)

    samples = np.vstack(params)
    summary = summarize_params(samples, base_names)
    out_path = args.iclv_dir / "bootstrap_param_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Saved: {out_path} (rows: {len(summary)})")


if __name__ == "__main__":
    main()
