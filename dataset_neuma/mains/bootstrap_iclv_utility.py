from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch

from src.models.icl_v import DeterministicICLV, param_names
from torch.nn.utils import parameters_to_vector
from utils.features import load_features_file


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
    df: pd.DataFrame, cols: Sequence[str], scaler: str = "standard", cat_unique_threshold: int = 50
):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

    df_prep = df[list(cols)].copy()
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


def encode_indicator_blocks(df_tr: pd.DataFrame, df_full: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    blocks = []
    for col in cols:
        if pd.api.types.is_numeric_dtype(df_tr[col]):
            med = df_tr[col].median()
            col_full = df_full[col].fillna(med)
        else:
            tr_str = df_tr[col].astype(str)
            uniq = tr_str.unique().tolist()
            mapping = {v: i for i, v in enumerate(uniq)}
            col_full = df_full[col].astype(str).map(mapping).fillna(-1)
        blocks.append(col_full.to_numpy(dtype=np.float32))
    if not blocks:
        return np.zeros((len(df_full), 0), dtype=np.float32)
    return np.stack(blocks, axis=1).astype(np.float32)


def filter_low_variance(X: np.ndarray, min_var: float) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[1] == 0:
        return X, np.ones(0, dtype=bool)
    var = np.var(X, axis=0)
    mask = var >= min_var
    return X[:, mask], mask


def bootstrap_indices(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(len(y)), size=n, replace=True)


def stars_for_t(t: float) -> str:
    if np.isnan(t):
        return ""
    if abs(t) >= 2.58:
        return "***"
    if abs(t) >= 1.96:
        return "**"
    if abs(t) >= 1.64:
        return "*"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap completo para ICLV tradicional.")
    parser.add_argument("--iclv-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--l2", type=float, default=0.0, help="Ridge L2 (global).")
    parser.add_argument("--by-subject", action="store_true", help="Bootstrap por sujeto.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_args = load_run_args(args.iclv_dir)
    data_path = args.data or Path(run_args.get("data", ""))
    if not data_path:
        raise ValueError("Se requiere --data o run_metadata.json con data.")

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()
    if "subject" not in df.columns and "id_sub" in df.columns:
        df["subject"] = df["id_sub"].astype(str)

    label_col = run_args.get("label_col", "bought").lower()
    if label_col not in df.columns:
        raise ValueError(f"No se encontro label '{label_col}'")

    # Apply same category collapsing
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

    obs_lt_cols = [c.strip().lower() for c in load_features_file(run_args.get("obs_lt_cols", ""))] if run_args.get("obs_lt_cols") else []
    obs_u_cols = [c.strip().lower() for c in load_features_file(run_args.get("obs_u_cols", ""))] if run_args.get("obs_u_cols") else []
    obs_i_cols = [c.strip().lower() for c in load_features_file(run_args.get("obs_i_cols", ""))] if run_args.get("obs_i_cols") else []
    obs_lt_cols = [c for c in obs_lt_cols if c in df.columns]
    obs_u_cols = [c for c in obs_u_cols if c in df.columns]
    obs_i_cols = [c for c in obs_i_cols if c in df.columns]

    train_subjects = set(load_split_subjects(args.iclv_dir))
    df_train = df[df["subject"].isin(train_subjects)].copy() if train_subjects else df

    scaler = run_args.get("scaler", "standard")
    cat_unique_threshold = int(run_args.get("cat_unique_threshold", 4))
    min_var = float(run_args.get("min_var", 1e-6))
    obs_u_buy_only = bool(run_args.get("obs_u_buy_only", False))

    X_lt_tr, preproc_lt = prepare_preprocessor(df_train, obs_lt_cols, scaler=scaler, cat_unique_threshold=cat_unique_threshold) if obs_lt_cols else (np.zeros((len(df_train), 0), dtype=np.float32), None)
    X_lt_all = to_float_array(preproc_lt.transform(df[obs_lt_cols].copy())) if obs_lt_cols else np.zeros((len(df), 0), dtype=np.float32)

    X_u_tr, preproc_u = prepare_preprocessor(df_train, obs_u_cols, scaler=scaler, cat_unique_threshold=cat_unique_threshold)
    X_u_all = to_float_array(preproc_u.transform(df[obs_u_cols].copy()))
    X_u_tr = to_float_array(X_u_tr)
    X_u_tr, mask = filter_low_variance(X_u_tr, min_var)
    if mask.size:
        X_u_all = X_u_all[:, mask]
    if obs_u_buy_only:
        X_u = np.zeros((len(df), int(run_args.get("num_choices", 2)), X_u_all.shape[1]), dtype=np.float32)
        X_u[:, 1, :] = X_u_all
    else:
        X_u = X_u_all[:, None, :].repeat(int(run_args.get("num_choices", 2)), axis=1)

    X_i = encode_indicator_blocks(df_train, df, obs_i_cols)
    y = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=np.int64)

    device = torch.device(args.device)
    n_choices = int(run_args.get("num_choices", 2))
    n_latent = int(run_args.get("n_latent", 1))
    model_init = DeterministicICLV(
        dim_obs_lt=X_lt_all.shape[1],
        dim_obs_u=X_u.shape[2],
        n_latent=n_latent,
        n_indicators=X_i.shape[1],
        n_choices=n_choices,
        delta_per_alt=bool(run_args.get("delta_per_alt", True)),
        beta_per_alt=bool(run_args.get("beta_per_alt", False)),
    ).to(device)

    param_names_list = param_names(model_init)
    param_samples = []

    for b in range(args.n_bootstrap):
        if args.by_subject and "subject" in df.columns:
            rng = np.random.default_rng(args.seed + b)
            uniq = df["subject"].unique()
            subs = rng.choice(uniq, size=len(uniq), replace=True)
            grouped = df.groupby("subject").indices
            idx_list = [grouped[s] for s in subs]
            idx = np.concatenate(idx_list)
        else:
            idx = bootstrap_indices(y, len(y), seed=args.seed + b)

        obs_lt_b = torch.tensor(X_lt_all[idx], dtype=torch.float32, device=device)
        obs_u_b = torch.tensor(X_u[idx], dtype=torch.float32, device=device)
        ind_b = torch.tensor(X_i[idx], dtype=torch.float32, device=device)
        y_b = torch.tensor(y[idx], dtype=torch.long, device=device)

        model = DeterministicICLV(
            dim_obs_lt=X_lt_all.shape[1],
            dim_obs_u=X_u.shape[2],
            n_latent=n_latent,
            n_indicators=X_i.shape[1],
            n_choices=n_choices,
            delta_per_alt=bool(run_args.get("delta_per_alt", True)),
            beta_per_alt=bool(run_args.get("beta_per_alt", False)),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(run_args.get("lr", 1e-3)) if args.lr is None else args.lr)

        for _ in range(args.max_iter):
            out = model(obs_lt_b, obs_u_b, ind_b, y_b)
            loss = out["loss"]
            if args.l2 > 0:
                l2_term = sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)
                loss = loss + args.l2 * l2_term
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        param_vec = parameters_to_vector([p for p in model.parameters() if p.requires_grad]).detach().cpu().numpy()
        param_samples.append(param_vec)

    params = np.vstack(param_samples)
    mean = params.mean(axis=0)
    std = params.std(axis=0, ddof=1)
    tstat = mean / np.where(std == 0, np.nan, std)

    rows = []
    for name, m, s, t in zip(param_names_list, mean, std, tstat):
        rows.append(
            {
                "name": name,
                "mean": float(m),
                "std": float(s),
                "tstat": float(t),
                "stars": stars_for_t(t),
            }
        )
    out_df = pd.DataFrame(rows)
    out_path = args.iclv_dir / "bootstrap_all_params.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} (rows: {len(out_df)})")


if __name__ == "__main__":
    main()
