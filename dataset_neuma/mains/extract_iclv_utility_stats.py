from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.features import load_features_file


def load_feature_names_from_preproc(preproc_path: Path, obs_u_cols: List[str]) -> List[str]:
    try:
        preproc = torch.load(preproc_path, weights_only=False)
        if hasattr(preproc, "get_feature_names_out"):
            return list(preproc.get_feature_names_out(obs_u_cols))
    except Exception:
        pass
    return obs_u_cols


def extract_beta_from_state(state: dict) -> Tuple[np.ndarray, int]:
    if "beta.weight" in state:
        beta = state["beta.weight"].detach().cpu().numpy()
        if beta.ndim == 2 and beta.shape[0] == 1:
            beta = beta[0:1, :]
        return beta, beta.shape[0]
    if "beta" in state:
        beta = state["beta"].detach().cpu().numpy()
        if beta.ndim == 1:
            beta = beta.reshape(1, -1)
        return beta, beta.shape[0]
    raise ValueError("No se encontro beta en state_dict.")


def load_run_args(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    return meta.get("args", {})


def load_split_subjects(run_dir: Path) -> List[str]:
    info = json.loads((run_dir / "split_info.json").read_text(encoding="utf-8"))
    return info.get("train_subjects", [])


def normalize_subject(val: str) -> str:
    s = str(val).strip().lower()
    digits = "".join([c for c in s if c.isdigit()])
    if digits:
        return str(int(digits))
    return s


def encode_indicators(train_df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    mats = []
    for col in cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            col_vals = train_df[col].fillna(train_df[col].median())
        else:
            tr_str = train_df[col].astype(str)
            uniq = tr_str.unique().tolist()
            mapping = {v: i for i, v in enumerate(uniq)}
            col_vals = tr_str.map(mapping).fillna(-1)
        mats.append(col_vals.to_numpy(dtype=np.float32))
    if not mats:
        return np.zeros((len(train_df), 0), dtype=np.float32)
    return np.stack(mats, axis=1).astype(np.float32)


def preprocess_block(train_df: pd.DataFrame, cols: List[str], prefix: str) -> Tuple[pd.DataFrame, List[str]]:
    import pandas.api.types as ptypes

    num_cols = [c for c in cols if ptypes.is_numeric_dtype(train_df[c])]
    cat_cols = [c for c in cols if c not in num_cols]

    out_parts = []
    names = []

    if num_cols:
        means = train_df[num_cols].mean()
        stds = train_df[num_cols].std().replace(0, 1)
        tr_num = (train_df[num_cols] - means) / stds
        names.extend([f"{prefix}{c}" for c in num_cols])
        tr_num.columns = names[: len(num_cols)]
        out_parts.append(tr_num)

    if cat_cols:
        tr_cat = pd.get_dummies(
            train_df[cat_cols].astype(str),
            prefix=[f"{prefix}{c}" for c in cat_cols],
            drop_first=True,
        )
        names.extend(tr_cat.columns.tolist())
        out_parts.append(tr_cat)

    if out_parts:
        tr_block = pd.concat(out_parts, axis=1)
    else:
        tr_block = pd.DataFrame(index=train_df.index)
    return tr_block, names


def opg_beta_stats(model: nn.Module, beta_param: torch.Tensor, batch_iter, max_rows: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    info = None
    n = 0
    for batch in batch_iter:
        if max_rows is not None and n >= max_rows:
            break
        obs_lt, obs_u, indicators, choice = batch
        obs_lt = obs_lt.to(beta_param.device)
        obs_u = obs_u.to(beta_param.device)
        indicators = indicators.to(beta_param.device)
        choice = choice.to(beta_param.device)
        B = obs_lt.shape[0]
        for i in range(B):
            if max_rows is not None and n >= max_rows:
                break
            out = model(obs_lt[i : i + 1], obs_u[i : i + 1], indicators[i : i + 1], choice[i : i + 1])
            logp_i = out["logp"][0, int(choice[i].item())]
            grad = torch.autograd.grad(logp_i, beta_param, retain_graph=False, create_graph=False)[0]
            g = grad.detach().flatten().double().cpu().numpy()
            if info is None:
                info = np.outer(g, g)
            else:
                info += np.outer(g, g)
            n += 1
    if info is None or n == 0:
        return np.array([]), np.array([])
    info = info / n
    try:
        var = np.linalg.pinv(info)
    except Exception:
        var = np.linalg.pinv(info + 1e-6 * np.eye(info.shape[0]))
    std = np.sqrt(np.clip(np.diag(var), 1e-12, None))
    theta = beta_param.detach().flatten().double().cpu().numpy()
    tstat = theta / std
    return std, tstat


def biogeme_beta_stats(
    model: nn.Module,
    beta_param: torch.Tensor,
    batch_iter_fn,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    model.eval()
    grads = []
    n = 0
    for batch in batch_iter_fn():
        if max_rows is not None and n >= max_rows:
            break
        obs_lt, obs_u, indicators, choice = batch
        obs_lt = obs_lt.to(beta_param.device)
        obs_u = obs_u.to(beta_param.device)
        indicators = indicators.to(beta_param.device)
        choice = choice.to(beta_param.device)
        B = obs_lt.shape[0]
        for i in range(B):
            if max_rows is not None and n >= max_rows:
                break
            out = model(obs_lt[i : i + 1], obs_u[i : i + 1], indicators[i : i + 1], choice[i : i + 1])
            logp_i = out["logp"][0, int(choice[i].item())]
            grad = torch.autograd.grad(logp_i, beta_param, retain_graph=False, create_graph=False)[0]
            grads.append(grad.detach().flatten().double())
            n += 1
    if not grads:
        return np.array([]), np.array([]), np.array([]), np.array([]), {}
    G = torch.stack(grads, dim=0)  # [N, P]
    info = (G.t() @ G).cpu().numpy()
    # Hessian of sum log-likelihood
    flat_init = beta_param.detach().flatten()

    def _loss_flat(flat_params: torch.Tensor) -> torch.Tensor:
        from torch.nn.utils import vector_to_parameters

        vector_to_parameters(flat_params, [beta_param])
        total = 0.0
        count = 0
        for obs_lt, obs_u, indicators, choice in batch_iter_fn():
            obs_lt = obs_lt.to(beta_param.device)
            obs_u = obs_u.to(beta_param.device)
            indicators = indicators.to(beta_param.device)
            choice = choice.to(beta_param.device)
            if max_rows is not None and count >= max_rows:
                break
            out = model(obs_lt, obs_u, indicators, choice)
            ll = out["logp"].gather(1, choice.view(-1, 1)).sum()
            total = total + ll
            count += obs_lt.size(0)
        return total if count > 0 else torch.tensor(0.0, device=beta_param.device)

    H = torch.autograd.functional.hessian(_loss_flat, flat_init).detach().cpu().numpy()
    # restore
    from torch.nn.utils import vector_to_parameters as _v2p

    _v2p(flat_init, [beta_param])
    # classic: (-A)^{-1}
    A = H
    eigvals = np.linalg.eigvalsh(-A) if A.size else np.array([])
    diag_info = {
        "lambda_min": float(np.min(eigvals)) if eigvals.size else np.nan,
        "lambda_max": float(np.max(eigvals)) if eigvals.size else np.nan,
        "cond": float(np.max(eigvals) / np.min(eigvals)) if eigvals.size and np.min(eigvals) != 0 else np.inf,
        "n_obs": int(n),
    }
    try:
        cov_classic = np.linalg.pinv(-A)
    except Exception:
        cov_classic = np.linalg.pinv(-A + 1e-6 * np.eye(A.shape[0]))
    std_classic = np.sqrt(np.clip(np.diag(cov_classic), 1e-12, None))
    tstat_classic = flat_init.detach().double().cpu().numpy() / std_classic
    # robust sandwich
    try:
        invA = np.linalg.pinv(-A)
    except Exception:
        invA = np.linalg.pinv(-A + 1e-6 * np.eye(A.shape[0]))
    cov_robust = invA @ info @ invA
    std_robust = np.sqrt(np.clip(np.diag(cov_robust), 1e-12, None))
    tstat_robust = flat_init.detach().double().cpu().numpy() / std_robust
    return std_classic, tstat_classic, std_robust, tstat_robust, diag_info


def load_hessian_stats(hessian_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    data = json.loads(hessian_path.read_text(encoding="utf-8"))
    theta = np.asarray(data["theta"])
    std = np.asarray(data["std"])
    names = list(data["names"])
    return theta, std, names


def stratified_sample_indices(y: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    if max_rows <= 0 or max_rows >= len(y):
        return np.arange(len(y))
    per_class = max_rows // len(classes)
    idxs = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        if len(c_idx) == 0:
            continue
        take = min(per_class, len(c_idx))
        idxs.append(rng.choice(c_idx, size=take, replace=False))
    if idxs:
        out = np.concatenate(idxs)
    else:
        out = rng.choice(np.arange(len(y)), size=max_rows, replace=False)
    if len(out) < max_rows:
        remaining = np.setdiff1d(np.arange(len(y)), out)
        if len(remaining) > 0:
            extra = rng.choice(remaining, size=min(max_rows - len(out), len(remaining)), replace=False)
            out = np.concatenate([out, extra])
    rng.shuffle(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae betas y estadisticos de utilidad ICLV.")
    parser.add_argument("--iclv-dir", type=Path, default=Path("./results/icl_v_standard/run_0001"))
    parser.add_argument("--mm-dir", type=Path, default=Path("./results/multimodal_icl_v_standard/run_0001"))
    parser.add_argument("--use-opg", action="store_true", help="Usar OPG/Fisher para std/tstat en betas.")
    parser.add_argument("--opg-max-rows", type=int, default=2000)
    parser.add_argument("--biogeme-stats", action="store_true", help="Calcula A/B (Biogeme) para betas.")
    parser.add_argument("--biogeme-max-rows", type=int, default=2000)
    parser.add_argument("--biogeme-seed", type=int, default=42)
    args = parser.parse_args()

    rows = []

    # ICLV clasico
    iclv_metrics = json.loads((args.iclv_dir / "metrics.json").read_text(encoding="utf-8"))
    obs_u_cols = iclv_metrics.get("obs_u_cols", [])
    feature_names = load_feature_names_from_preproc(args.iclv_dir / "preproc_u.pkl", obs_u_cols)
    state = torch.load(args.iclv_dir / "model.pt", map_location="cpu", weights_only=True)
    beta, n_alts = extract_beta_from_state(state)
    mnl_only = bool(load_run_args(args.iclv_dir).get("mnl_only", False))
    n_alts_loop = beta.shape[0]

    opg_std = opg_t = None
    biog_std = biog_t = biog_std_r = biog_t_r = None
    biog_diag = {}
    if args.use_opg:
        run_args = load_run_args(args.iclv_dir)
        data_path = Path(run_args.get("data", ""))
        obs_lt_file = run_args.get("obs_lt_cols")
        obs_u_file = run_args.get("obs_u_cols")
        obs_i_file = run_args.get("obs_i_cols")
        if data_path and data_path.exists():
            df = pd.read_csv(data_path)
            df.columns = df.columns.str.lower()
            if "subject" not in df.columns and "id_sub" in df.columns:
                df["subject"] = df["id_sub"].astype(str)
            train_subjects = set(load_split_subjects(args.iclv_dir))
            df = df[df["subject"].isin(train_subjects)].reset_index(drop=True)
            obs_lt_cols_use = [c.strip().lower() for c in (load_features_file(obs_lt_file) if obs_lt_file else [])]
            obs_u_cols_use = [c.strip().lower() for c in (load_features_file(obs_u_file) if obs_u_file else [])]
            obs_i_cols_use = [c.strip().lower() for c in (load_features_file(obs_i_file) if obs_i_file else [])]
            preproc_lt = torch.load(args.iclv_dir / "preproc_lt.pkl", weights_only=False)
            preproc_u = torch.load(args.iclv_dir / "preproc_u.pkl", weights_only=False)
            if preproc_lt is None or not obs_lt_cols_use:
                X_lt = np.zeros((len(df), 0), dtype=np.float32)
            else:
                X_lt = preproc_lt.transform(df[obs_lt_cols_use].copy())
            X_u = preproc_u.transform(df[obs_u_cols_use].copy())
            ind = encode_indicators(df, obs_i_cols_use)
            y = pd.to_numeric(df[iclv_metrics.get("label_col", "bought")], errors="coerce").to_numpy(dtype=np.int64) if "label_col" in iclv_metrics else pd.to_numeric(df["bought"], errors="coerce").to_numpy(dtype=np.int64)
            # build tensors
            obs_lt_t = torch.tensor(X_lt, dtype=torch.float64)
            obs_u_t = torch.tensor(X_u, dtype=torch.float64)
            mnl_only = bool(run_args.get("mnl_only", False))
            if mnl_only:
                n_alts = int(run_args.get("num_choices", df["bought"].nunique()))
            if obs_u_t.dim() == 2:
                obs_u_t = obs_u_t.unsqueeze(1).expand(-1, n_alts, -1)
            ind_t = torch.tensor(ind, dtype=torch.float64)
            y_t = torch.tensor(y, dtype=torch.long)
            # rebuild model
            if mnl_only:
                class SimpleMNL(nn.Module):
                    def __init__(self, dim_obs_u: int, n_choices: int):
                        super().__init__()
                        self.beta = nn.Linear(dim_obs_u, 1, bias=False)
                        self.ASC = nn.Parameter(torch.zeros(n_choices))

                    def forward(self, obs_lt, obs_u, indicators, choice):
                        if obs_u.dim() == 2:
                            obs_u = obs_u.unsqueeze(1).expand(-1, self.ASC.numel(), -1)
                        logp = F.log_softmax(self.beta(obs_u).squeeze(-1) + self.ASC.unsqueeze(0), dim=1)
                        ll = logp.gather(1, choice.view(-1, 1)).sum()
                        return {"logp": logp, "log_likelihood": ll}

                model = SimpleMNL(dim_obs_u=obs_u_t.shape[2], n_choices=n_alts)
                # map beta.weight key from checkpoint
                if "beta.weight" in state and "beta" not in state:
                    state = dict(state)
                    state["beta.weight"] = state.pop("beta.weight")
                # handle ASC shape mismatch (old runs with base alt)
                if "ASC" in state and state["ASC"].numel() != model.ASC.numel():
                    state = dict(state)
                    state.pop("ASC", None)
                model.load_state_dict(state, strict=False)
            else:
                from src.models.icl_v import DeterministicICLV

                model = DeterministicICLV(
                    dim_obs_lt=obs_lt_t.shape[1],
                    dim_obs_u=obs_u_t.shape[2],
                    n_latent=int(iclv_metrics.get("n_latent", 3)),
                    n_indicators=ind_t.shape[1],
                    n_choices=n_alts,
                    alpha=1.0,
                    delta_per_alt=True,
                    beta_per_alt=("beta[" in " ".join(json.loads((args.iclv_dir / "hessian.json").read_text(encoding="utf-8"))["names"]) if (args.iclv_dir / "hessian.json").exists() else False),
                )
                model.load_state_dict(state, strict=False)
            model = model.double()
            beta_param = model.beta if isinstance(model.beta, torch.nn.Parameter) else model.beta.weight
            # iterate in batches
            def batch_iter(bs=32):
                for i in range(0, len(obs_lt_t), bs):
                    yield (obs_lt_t[i : i + bs], obs_u_t[i : i + bs], ind_t[i : i + bs], y_t[i : i + bs])
            opg_std, opg_t = opg_beta_stats(model, beta_param, batch_iter(), max_rows=args.opg_max_rows)
            if args.biogeme_stats:
                biog_std, biog_t, biog_std_r, biog_t_r, biog_diag = biogeme_beta_stats(
                    model, beta_param, batch_iter, max_rows=args.biogeme_max_rows
                )

    if (args.iclv_dir / "hessian.json").exists() and not args.use_opg and not args.biogeme_stats:
        theta, std, names = load_hessian_stats(args.iclv_dir / "hessian.json")
        beta_idx = [i for i, n in enumerate(names) if n.startswith("beta")]
        for alt in range(n_alts_loop):
            for j, feat in enumerate(feature_names):
                idx = alt * beta.shape[1] + j
                if idx >= beta.shape[1] * n_alts:
                    continue
                flat_idx = beta_idx[idx] if idx < len(beta_idx) else None
                coef = beta[alt, j]
                sd = std[flat_idx] if (flat_idx is not None and flat_idx < len(std)) else np.nan
                tstat = coef / sd if sd == sd and sd != 0 else np.nan
                rows.append(
                    {
                        "model": "icl_v",
                        "alt": alt,
                        "feature": feat,
                        "coef": float(coef),
                        "std": float(sd) if sd == sd else np.nan,
                        "tstat": float(tstat) if tstat == tstat else np.nan,
                        "method": "hessian",
                    }
                )
    elif args.use_opg and opg_std is not None and opg_std.size:
        for alt in range(n_alts_loop):
            for j, feat in enumerate(feature_names):
                idx = alt * beta.shape[1] + j
                coef = beta[alt, j]
                sd = opg_std[idx] if idx < len(opg_std) else np.nan
                tstat = opg_t[idx] if idx < len(opg_t) else np.nan
                rows.append(
                    {
                        "model": "icl_v",
                        "alt": alt,
                        "feature": feat,
                        "coef": float(coef),
                        "std": float(sd) if sd == sd else np.nan,
                        "tstat": float(tstat) if tstat == tstat else np.nan,
                        "method": "opg",
                    }
                )
        if biog_diag:
            print(f"[biogeme] iclv lambda_min={biog_diag.get('lambda_min')} lambda_max={biog_diag.get('lambda_max')} cond={biog_diag.get('cond')}")
    elif args.biogeme_stats and biog_std is not None and biog_std.size:
        for alt in range(n_alts_loop):
            for j, feat in enumerate(feature_names):
                idx = alt * beta.shape[1] + j
                coef = beta[alt, j]
                rows.append(
                    {
                        "model": "icl_v",
                        "alt": alt,
                        "feature": feat,
                        "coef": float(coef),
                        "std": float(biog_std[idx]),
                        "tstat": float(biog_t[idx]),
                        "std_robust": float(biog_std_r[idx]),
                        "tstat_robust": float(biog_t_r[idx]),
                        "method": "biogeme",
                    }
                )
        if biog_diag:
            print(f"[biogeme] iclv lambda_min={biog_diag.get('lambda_min')} lambda_max={biog_diag.get('lambda_max')} cond={biog_diag.get('cond')}")
    else:
        for alt in range(n_alts):
            for j, feat in enumerate(feature_names):
                rows.append(
                    {
                        "model": "icl_v",
                        "alt": alt,
                        "feature": feat,
                        "coef": float(beta[alt, j]),
                        "std": np.nan,
                        "tstat": np.nan,
                    }
                )

    # ICLV multimodal
    mm_metrics = json.loads((args.mm_dir / "metrics.json").read_text(encoding="utf-8"))
    mm_obs_u_cols = mm_metrics.get("obs_u_cols", [])
    state_mm = torch.load(args.mm_dir / "model_last.pt", map_location="cpu", weights_only=True)
    beta_mm, n_alts_mm = extract_beta_from_state(state_mm)
    opg_std_mm = opg_t_mm = None
    biog_std_mm = biog_t_mm = biog_std_r_mm = biog_t_r_mm = None
    biog_diag_mm = {}
    if args.use_opg:
        run_args = load_run_args(args.mm_dir)
        data_path = Path(run_args.get("data", ""))
        obs_lt_file = run_args.get("obs_lt_cols")
        obs_u_file = run_args.get("obs_u_cols")
        if data_path and data_path.exists():
            df = pd.read_csv(data_path)
            df.columns = df.columns.str.lower()
            if "subject" not in df.columns and "id_sub" in df.columns:
                df["subject"] = df["id_sub"].astype(str)
            train_subjects = set(load_split_subjects(args.mm_dir))
            df = df[df["subject"].isin(train_subjects)].reset_index(drop=True)
            # imputaciones clave
            for c in ["gender", "maritalstatus", "supermarketvisitduration", "shoppinglist", "offer"]:
                if c in df.columns:
                    mode = df[c].mode(dropna=True)
                    df[c] = df[c].fillna(mode.iloc[0] if len(mode) else "missing")
            for c in ["price", "len_med"]:
                if c in df.columns:
                    df[c] = df[c].fillna(df[c].median())
            df = df.dropna(subset=[run_args.get("label_col", "bought"), run_args.get("img_emb_col", "embedding_path"), run_args.get("eeg_emb_col", "eeg_emb_path")])
            obs_lt_cols_use = [c.strip().lower() for c in (load_features_file(obs_lt_file) if obs_lt_file else [])]
            obs_u_cols_use = [c.strip().lower() for c in (load_features_file(obs_u_file) if obs_u_file else [])]
            lt_block, lt_names = preprocess_block(df, obs_lt_cols_use, prefix="lt_")
            u_block, u_names = preprocess_block(df, obs_u_cols_use, prefix="u_")
            obs_lt = torch.tensor(lt_block.to_numpy(dtype=np.float64), dtype=torch.float64)
            obs_u = torch.tensor(u_block.to_numpy(dtype=np.float64), dtype=torch.float64)
            if obs_u.dim() == 2:
                obs_u = obs_u.unsqueeze(1).expand(-1, n_alts_mm, -1)
            # load embeddings
            img_paths = df[run_args.get("img_emb_col", "embedding_path")].tolist()
            eeg_paths = df[run_args.get("eeg_emb_col", "eeg_emb_path")].tolist()
            img_emb = torch.tensor(np.stack([np.load(p).flatten() for p in img_paths]).astype(np.float64))
            eeg_emb = torch.tensor(np.stack([np.load(p).flatten() for p in eeg_paths]).astype(np.float64))
            y = pd.to_numeric(df[run_args.get("label_col", "bought")], errors="coerce").to_numpy(dtype=np.int64)
            y_t = torch.tensor(y, dtype=torch.long)
            from src.models.multimodal_icl_v import MultimodalICLVDeterministic
            beta_is_per_alt = False
            beta_key = "beta.weight" if "beta.weight" in state_mm else "beta" if "beta" in state_mm else None
            if beta_key is not None:
                beta_shape = tuple(state_mm[beta_key].shape)
                if len(beta_shape) == 2 and beta_shape[0] == n_alts_mm:
                    beta_is_per_alt = True
            model = MultimodalICLVDeterministic(
                dim_obs_lt=obs_lt.shape[1],
                dim_obs_u=obs_u.shape[2],
                dim_img_emb=img_emb.shape[1],
                dim_eeg_emb=eeg_emb.shape[1],
                n_latent=int(mm_metrics.get("n_latent", 3)),
                n_choices=n_alts_mm,
                alpha=1.0,
                img_proj_dim=int(mm_metrics.get("img_proj_dim", 32)),
                beta_per_alt=beta_is_per_alt,
            )
            # handle beta param name differences between Linear vs Parameter
            if "beta" in state_mm and "beta.weight" not in state_mm:
                state_mm = dict(state_mm)
                state_mm["beta.weight"] = state_mm.pop("beta")
            model.load_state_dict(state_mm, strict=False)
            model = model.double()
            beta_param = model.beta if isinstance(model.beta, torch.nn.Parameter) else model.beta.weight
            # batch iter
            def batch_iter(bs=32):
                for i in range(0, len(obs_lt), bs):
                    yield (obs_lt[i : i + bs], obs_u[i : i + bs], eeg_emb[i : i + bs], img_emb[i : i + bs], y_t[i : i + bs])

            # adapt opg function to multimodal inputs
            info = None
            n = 0
            for obs_lt_b, obs_u_b, eeg_b, img_b, y_b in batch_iter():
                if args.opg_max_rows and n >= args.opg_max_rows:
                    break
                for i in range(obs_lt_b.shape[0]):
                    if args.opg_max_rows and n >= args.opg_max_rows:
                        break
                    out = model(obs_lt_b[i : i + 1], obs_u_b[i : i + 1], eeg_b[i : i + 1], img_b[i : i + 1], y_b[i : i + 1])
                    logp_i = out["logp"][0, int(y_b[i].item())]
                    grad = torch.autograd.grad(logp_i, beta_param, retain_graph=False, create_graph=False)[0]
                    g = grad.detach().flatten().double().cpu().numpy()
                    if info is None:
                        info = np.outer(g, g)
                    else:
                        info += np.outer(g, g)
                    n += 1
            if info is not None and n > 0:
                info = info / n
                var = np.linalg.pinv(info)
                opg_std_mm = np.sqrt(np.clip(np.diag(var), 1e-12, None))
                theta = beta_param.detach().flatten().double().cpu().numpy()
                opg_t_mm = theta / opg_std_mm
            if args.biogeme_stats:
                # build iterator for biogeme stats (stratified sample if requested)
                obs_lt_b = obs_lt
                obs_u_b = obs_u
                eeg_b = eeg_emb
                img_b = img_emb
                y_b = y_t
                if args.biogeme_max_rows and args.biogeme_max_rows < len(y):
                    idx = stratified_sample_indices(y, args.biogeme_max_rows, args.biogeme_seed)
                    obs_lt_b = obs_lt_b[idx]
                    obs_u_b = obs_u_b[idx]
                    eeg_b = eeg_b[idx]
                    img_b = img_b[idx]
                    y_b = y_b[idx]

                def batch_iter_b(bs=32):
                    for i in range(0, len(obs_lt_b), bs):
                        yield (obs_lt_b[i : i + bs], obs_u_b[i : i + bs], eeg_b[i : i + bs], img_b[i : i + bs], y_b[i : i + bs])

                biog_std_mm, biog_t_mm, biog_std_r_mm, biog_t_r_mm, biog_diag_mm = biogeme_beta_stats(
                    model, beta_param, batch_iter_b, max_rows=None
                )

    if (args.mm_dir / "hessian.json").exists() and not args.use_opg and not args.biogeme_stats:
        theta, std, names = load_hessian_stats(args.mm_dir / "hessian.json")
        beta_idx = [i for i, n in enumerate(names) if n.startswith("beta")]
        for alt in range(n_alts_mm):
            for j, feat in enumerate(mm_obs_u_cols):
                idx = alt * beta_mm.shape[1] + j
                flat_idx = beta_idx[idx] if idx < len(beta_idx) else None
                coef = beta_mm[alt, j]
                sd = std[flat_idx] if (flat_idx is not None and flat_idx < len(std)) else np.nan
                tstat = coef / sd if sd == sd and sd != 0 else np.nan
                rows.append(
                    {
                        "model": "multimodal_icl_v",
                        "alt": alt,
                        "feature": feat,
                        "coef": float(coef),
                        "std": float(sd) if sd == sd else np.nan,
                        "tstat": float(tstat) if tstat == tstat else np.nan,
                        "method": "hessian",
                    }
                )
    elif args.use_opg and opg_std_mm is not None and opg_std_mm.size:
        for alt in range(n_alts_mm):
            for j, feat in enumerate(mm_obs_u_cols):
                idx = alt * beta_mm.shape[1] + j
                coef = beta_mm[alt, j]
                sd = opg_std_mm[idx] if idx < len(opg_std_mm) else np.nan
                tstat = opg_t_mm[idx] if idx < len(opg_t_mm) else np.nan
                rows.append(
                    {
                        "model": "multimodal_icl_v",
                        "alt": alt,
                        "feature": feat,
                        "coef": float(coef),
                        "std": float(sd) if sd == sd else np.nan,
                        "tstat": float(tstat) if tstat == tstat else np.nan,
                        "method": "opg",
                    }
                )
        if biog_diag_mm:
            print(f"[biogeme] mm lambda_min={biog_diag_mm.get('lambda_min')} lambda_max={biog_diag_mm.get('lambda_max')} cond={biog_diag_mm.get('cond')}")
    elif args.biogeme_stats and biog_std_mm is not None and biog_std_mm.size:
        for alt in range(n_alts_mm):
            for j, feat in enumerate(mm_obs_u_cols):
                idx = alt * beta_mm.shape[1] + j
                coef = beta_mm[alt, j]
                rows.append(
                    {
                        "model": "multimodal_icl_v",
                        "alt": alt,
                        "feature": feat,
                        "coef": float(coef),
                        "std": float(biog_std_mm[idx]),
                        "tstat": float(biog_t_mm[idx]),
                        "std_robust": float(biog_std_r_mm[idx]),
                        "tstat_robust": float(biog_t_r_mm[idx]),
                        "method": "biogeme",
                    }
                )
        if biog_diag_mm:
            print(f"[biogeme] mm lambda_min={biog_diag_mm.get('lambda_min')} lambda_max={biog_diag_mm.get('lambda_max')} cond={biog_diag_mm.get('cond')}")
    else:
        for alt in range(n_alts_mm):
            for j, feat in enumerate(mm_obs_u_cols):
                rows.append(
                    {
                        "model": "multimodal_icl_v",
                        "alt": alt,
                        "feature": feat,
                        "coef": float(beta_mm[alt, j]),
                        "std": np.nan,
                        "tstat": np.nan,
                    }
                )

    out_df = pd.DataFrame(rows)

    # Effects: compra (alt=1) vs no compra (alt=0), sin covarianza entre betas
    effects_rows = []
    for model_name, beta_mat, feats, std_map in [
        ("icl_v", beta, feature_names, None),
        ("multimodal_icl_v", beta_mm, mm_obs_u_cols, None),
    ]:
        if beta_mat.shape[0] < 2:
            continue
        for j, feat in enumerate(feats):
            b1 = beta_mat[1, j]
            b0 = beta_mat[0, j]
            diff = b1 - b0
            # std aproximada sin cov: sqrt(std1^2 + std0^2)
            std1 = std0 = np.nan
            if not out_df.empty:
                rows_m = out_df[(out_df["model"] == model_name) & (out_df["feature"] == feat)]
                if not rows_m.empty:
                    stds = rows_m.sort_values("alt")["std"].to_numpy()
                    if len(stds) >= 2:
                        std0, std1 = stds[0], stds[1]
            if np.isfinite(std0) and np.isfinite(std1):
                std_eff = np.sqrt(std0**2 + std1**2)
                t_eff = diff / std_eff if std_eff > 0 else np.nan
            else:
                std_eff = np.nan
                t_eff = np.nan
            effects_rows.append(
                {
                    "model": model_name,
                    "feature": feat,
                    "effect_alt1_vs_alt0": float(diff),
                    "std_approx": float(std_eff) if std_eff == std_eff else np.nan,
                    "tstat_approx": float(t_eff) if t_eff == t_eff else np.nan,
                    "base_alt": 0,
                }
            )
    iclv_out = args.iclv_dir / "utility_stats.csv"
    mm_out = args.mm_dir / "utility_stats.csv"
    out_df[out_df["model"] == "icl_v"].to_csv(iclv_out, index=False)
    out_df[out_df["model"] == "multimodal_icl_v"].to_csv(mm_out, index=False)
    if effects_rows:
        eff_df = pd.DataFrame(effects_rows)
        eff_df[eff_df["model"] == "icl_v"].to_csv(args.iclv_dir / "utility_effects.csv", index=False)
        eff_df[eff_df["model"] == "multimodal_icl_v"].to_csv(args.mm_dir / "utility_effects.csv", index=False)
        print("Saved effects:", args.iclv_dir / "utility_effects.csv")
        print("Saved effects:", args.mm_dir / "utility_effects.csv")
    print(f"Saved: {iclv_out} (rows: {len(out_df[out_df['model']=='icl_v'])})")
    print(f"Saved: {mm_out} (rows: {len(out_df[out_df['model']=='multimodal_icl_v'])})")


if __name__ == "__main__":
    main()
