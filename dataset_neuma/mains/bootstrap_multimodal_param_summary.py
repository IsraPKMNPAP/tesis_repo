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
    mean = np.nanmean(samples, axis=0)
    median = np.nanmedian(samples, axis=0)
    mean_abs = np.nanmean(np.abs(samples), axis=0)
    median_abs = np.nanmedian(np.abs(samples), axis=0)
    p25 = np.nanpercentile(samples, 25, axis=0)
    p75 = np.nanpercentile(samples, 75, axis=0)
    sign = np.sign(median)
    sign_pct = np.nanmean(np.sign(samples) == np.sign(median), axis=0)
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


def parse_flat_index(name: str) -> int | None:
    if "[" not in name or "]" not in name:
        return None
    try:
        idx = name.rsplit("[", 1)[-1].split("]")[0]
        return int(idx)
    except Exception:
        return None


def build_param_annotations(
    names: List[str],
    u_names: List[str],
    lt_names: List[str],
    img_proj_dim: int,
    n_choices: int,
    beta_per_alt: bool,
) -> pd.DataFrame:
    base_alt = 0
    alt_list = [a for a in range(n_choices) if a != base_alt]
    gamma0_inputs = lt_names + [f"img_proj_{i}" for i in range(img_proj_dim)]
    rows = []
    for name in names:
        block = "other"
        var_name = name
        alt = None
        if name.startswith("beta"):
            block = "utility"
            idx = parse_flat_index(name)
            if idx is not None and u_names:
                if beta_per_alt:
                    dim_u = len(u_names)
                    alt_pos = idx // dim_u
                    feat_idx = idx % dim_u
                    if alt_pos < len(alt_list):
                        alt = alt_list[alt_pos]
                    var_name = u_names[feat_idx] if feat_idx < len(u_names) else var_name
                else:
                    var_name = u_names[idx] if idx < len(u_names) else var_name
        elif name.startswith("Gamma.0.weight"):
            block = "structural"
            idx = parse_flat_index(name)
            if idx is not None and gamma0_inputs:
                in_dim = len(gamma0_inputs)
                col = idx % in_dim
                var_name = gamma0_inputs[col] if col < in_dim else var_name
        elif name.startswith("Gamma.0.bias"):
            block = "structural"
            var_name = "intercept"
        rows.append({"name": name, "block": block, "var_name": var_name, "alt": alt})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap multimodal: resumen de coeficientes.")
    parser.add_argument("--iclv-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lbfgs-steps", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--by-subject", action="store_true", help="Bootstrap por sujeto.")
    parser.add_argument("--beta-only", action="store_true", help="Resumir solo betas de utilidad.")
    parser.add_argument(
        "--tabular-only",
        action="store_true",
        help="Resumir solo parametros ligados a bloques tabulares (beta/Gamma).",
    )
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
    ckpt_path = args.iclv_dir / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = args.iclv_dir / "model_last.pt"
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)
        base_model.load_state_dict(ckpt, strict=False)
    base_model.eval()
    base_params, base_names = expand_param_names(base_model)
    print(
        f"[bootstrap] beta_per_alt={model_params['beta_per_alt']} n_latent={model_params['n_latent']} "
        f"obs_u_buy_only={obs_u_buy_only}"
    )

    params = []
    rng = np.random.default_rng(args.seed)
    grouped = df.groupby("subject").indices if "subject" in df.columns else {}
    unique_subjects = df["subject"].unique() if "subject" in df.columns else []
    for b in range(args.n_bootstrap):
        if args.by_subject and len(unique_subjects) > 0:
            subs = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
            idx_list = [grouped[s] for s in subs]
            idx = np.concatenate(idx_list)
        else:
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
        model.load_state_dict(base_model.state_dict(), strict=False)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_ll = -np.inf
        patience = 0
        for _ in range(args.max_iter):
            out = model(obs_lt_b, obs_u_b, eeg_b, img_b, y_b)
            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            ll = float(out["log_likelihood"].item())
            if ll > best_ll + 1e-6:
                best_ll = ll
                patience = 0
            else:
                patience += 1
                if patience >= args.early_stop_patience:
                    break

        # Full-batch LBFGS refinement
        if args.lbfgs_steps > 0:
            lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=args.lbfgs_steps, line_search_fn="strong_wolfe")

            def closure():
                lbfgs.zero_grad()
                out_lb = model(obs_lt_b, obs_u_b, eeg_b, img_b, y_b)
                loss_lb = out_lb["loss"]
                loss_lb.backward()
                return loss_lb

            lbfgs.step(closure)

        params_list, _ = expand_param_names(model)
        vec = torch.nn.utils.parameters_to_vector(params_list).detach().cpu().numpy()
        if not np.isfinite(vec).all():
            continue
        params.append(vec)

    if not params:
        raise ValueError("Bootstrap vacio: todas las replicas devolvieron NaN/Inf.")
    samples = np.vstack(params)
    keep = np.isfinite(samples).all(axis=1)
    if not np.all(keep):
        dropped = int(np.sum(~keep))
        print(f"[bootstrap][warn] dropped {dropped} replicas con NaN/Inf.")
        samples = samples[keep]
    if args.beta_only and args.tabular_only:
        raise ValueError("--beta-only y --tabular-only son excluyentes.")
    if args.beta_only:
        keep_idx = [i for i, n in enumerate(base_names) if n.startswith("beta")]
        samples = samples[:, keep_idx]
        base_names = [base_names[i] for i in keep_idx]
    elif args.tabular_only:
        keep_idx = [i for i, n in enumerate(base_names) if n.startswith("beta") or n.startswith("Gamma")]
        samples = samples[:, keep_idx]
        base_names = [base_names[i] for i in keep_idx]
    summary = summarize_params(samples, base_names)
    annotations = build_param_annotations(
        base_names,
        u_names,
        lt_names,
        int(run_args.get("img_proj_dim", 32)),
        n_choices,
        bool(run_args.get("beta_per_alt", False)),
    )
    summary = annotations.merge(summary, on="name", how="left")
    out_path = args.iclv_dir / "bootstrap_param_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Saved: {out_path} (rows: {len(summary)})")


if __name__ == "__main__":
    main()
