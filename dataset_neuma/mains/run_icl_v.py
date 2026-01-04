from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from torch.utils.data import DataLoader

# Imports locales
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading.icl_v import ICLVDataset
from src.models.icl_v import DeterministicICLV, compute_hessian_stats
from utils.features import load_features_file


def split_train_val(df: pd.DataFrame, label_col: str, val_split: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    if val_split <= 0 or val_split >= 1:
        return df.reset_index(drop=True), df.iloc[0:0].copy()
    labels = pd.to_numeric(df[label_col], errors="coerce")
    uniq = labels.dropna().unique()
    val_idx: List[int] = []
    for c in uniq:
        idx = np.where(labels == c)[0]
        k = int(max(1, round(len(idx) * val_split)))
        val_idx.extend(rng.choice(idx, size=min(k, len(idx)), replace=False))
    val_idx = sorted(set(val_idx))
    mask = np.zeros(len(df), dtype=bool)
    mask[val_idx] = True
    df_val = df.iloc[mask].reset_index(drop=True)
    df_tr = df.iloc[~mask].reset_index(drop=True)
    return df_tr, df_val


def to_float_array(mat) -> np.ndarray:
    try:
        arr = mat.toarray()
    except Exception:
        arr = np.asarray(mat)
    return arr.astype(np.float32)


def prepare_preprocessor(df: pd.DataFrame, cols: Sequence[str], scaler: str = "standard"):
    df_prep = df[cols].copy()
    # Inferir tipos: convertimos strings a categoría
    for c in df_prep.columns:
        if df_prep[c].dtype == object:
            df_prep[c] = df_prep[c].astype("category")
    numeric = df_prep.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical = df_prep.select_dtypes(include=["category"]).columns.tolist()
    scaler_cls = RobustScaler if scaler == "robust" else StandardScaler
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler_cls(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    mat = preprocessor.fit_transform(df_prep)
    return mat, preprocessor


def encode_indicator_blocks(df_tr: pd.DataFrame, df_val: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Convierte indicadores mixtos a numérico (factoriza strings/categorías)."""
    tr_blocks = []
    val_blocks = []
    for col in cols:
        if pd.api.types.is_numeric_dtype(df_tr[col]):
            tr_col = df_tr[col].fillna(df_tr[col].median())
            val_col = df_val[col].fillna(df_tr[col].median())
        else:
            tr_str = df_tr[col].astype(str)
            uniq = tr_str.unique().tolist()
            mapping = {v: i for i, v in enumerate(uniq)}
            tr_col = tr_str.map(mapping).fillna(-1)
            val_col = df_val[col].astype(str).map(mapping).fillna(-1)
        tr_blocks.append(tr_col.to_numpy(dtype=np.float32))
        val_blocks.append(val_col.to_numpy(dtype=np.float32))
    if not tr_blocks:
        return np.zeros((len(df_tr), 0), dtype=np.float32), np.zeros((len(df_val), 0), dtype=np.float32)
    tr_mat = np.stack(tr_blocks, axis=1).astype(np.float32)
    val_mat = np.stack(val_blocks, axis=1).astype(np.float32)
    return tr_mat, val_mat


def resolve_cols(df: pd.DataFrame, cols_file: str | None, fallback_numeric: bool, drop_cols: set) -> List[str]:
    """Si cols_file está definido, se usa; normaliza a minúsculas para compatibilidad."""
    if cols_file:
        cols = [c.strip().lower() for c in load_features_file(cols_file)]
    else:
        cols = []
    if not cols and fallback_numeric:
        cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if c in df.columns]
    return cols


def build_datasets(
    df_tr: pd.DataFrame,
    df_val: pd.DataFrame,
    obs_lt_cols: Sequence[str],
    obs_u_cols: Sequence[str],
    indicator_cols: Sequence[str],
    label_col: str,
    num_choices: int,
    scaler: str = "standard",
):
    X_lt_tr, preproc_lt = prepare_preprocessor(df_tr, obs_lt_cols, scaler=scaler)
    X_lt_val = preproc_lt.transform(df_val[obs_lt_cols].copy())

    X_u_tr, preproc_u = prepare_preprocessor(df_tr, obs_u_cols, scaler=scaler)
    X_u_val = preproc_u.transform(df_val[obs_u_cols].copy())

    if indicator_cols:
        ind_tr_mat, ind_val_mat = encode_indicator_blocks(df_tr[indicator_cols].copy(), df_val[indicator_cols].copy(), indicator_cols)
    else:
        ind_tr_mat = np.zeros((len(df_tr), 0), dtype=np.float32)
        ind_val_mat = np.zeros((len(df_val), 0), dtype=np.float32)

    y_tr = pd.to_numeric(df_tr[label_col], errors="coerce").to_numpy(dtype=np.int64)
    y_val = pd.to_numeric(df_val[label_col], errors="coerce").to_numpy(dtype=np.int64)

    train_ds = ICLVDataset(
        obs_lt=to_float_array(X_lt_tr),
        obs_u=to_float_array(X_u_tr),
        indicators=ind_tr_mat,
        choices=y_tr,
        num_choices=num_choices,
    )
    val_ds = ICLVDataset(
        obs_lt=to_float_array(X_lt_val),
        obs_u=to_float_array(X_u_val),
        indicators=ind_val_mat,
        choices=y_val,
        num_choices=num_choices,
    )
    return train_ds, val_ds, preproc_lt, preproc_u


def run_epoch(model, loader, device, train: bool = True, optimizer=None):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_choice = 0.0
    total_meas = 0.0
    total_ll = 0.0
    correct = 0
    total = 0
    y_true_all = []
    y_pred_all = []
    for obs_lt, obs_u, indicators, choice in loader:
        obs_lt = obs_lt.to(device)
        obs_u = obs_u.to(device)
        indicators = indicators.to(device)
        choice_t = torch.as_tensor(choice, device=device, dtype=torch.long)

        out = model(obs_lt, obs_u, indicators, choice_t)
        loss = out["loss"]
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * obs_lt.size(0)
        total_choice += float(out["loss_choice"].item()) * obs_lt.size(0)
        total_meas += float(out["loss_meas"].item()) * obs_lt.size(0)
        total_ll += float(out["log_likelihood"].item())
        preds = out["logp"].argmax(dim=1)
        correct += int((preds == choice_t).sum().item())
        total += obs_lt.size(0)
        y_true_all.append(choice_t.detach().cpu())
        y_pred_all.append(preds.detach().cpu())

    avg_loss = total_loss / max(1, total)
    avg_choice = total_choice / max(1, total)
    avg_meas = total_meas / max(1, total)
    if y_true_all:
        y_true_cat = torch.cat(y_true_all).numpy()
        y_pred_cat = torch.cat(y_pred_all).numpy()
    else:
        y_true_cat = np.array([])
        y_pred_cat = np.array([])
    acc = correct / max(1, total)
    return {
        "loss": avg_loss,
        "loss_choice": avg_choice,
        "loss_meas": avg_meas,
        "acc": acc,
        "log_likelihood": total_ll,
        "n": total,
        "y_true": y_true_cat,
        "y_pred": y_pred_cat,
    }


def main():
    parser = argparse.ArgumentParser(description="ICLV determinista clásico para NEUMA.")
    parser.add_argument("--data", type=Path, default=Path("./data/processed/multimodal_join.csv"), help="CSV con las observaciones.")
    parser.add_argument("--label-col", type=str, default="bought")
    parser.add_argument("--obs-lt-cols", type=str, default=None, help="Archivo txt con columnas OBS_LT.")
    parser.add_argument("--obs-u-cols", type=str, default=None, help="Archivo txt con columnas OBS_U.")
    parser.add_argument("--obs-i-cols", type=str, default=None, help="Archivo txt con columnas indicadores.")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0, help="Peso del bloque de medición.")
    parser.add_argument("--n-latent", type=int, default=3)
    parser.add_argument("--delta-per-alt", action="store_true", help="Si se usa delta específico por alternativa.")
    parser.add_argument("--num-choices", type=int, default=2, help="Número de alternativas (para compra sí/no = 2).")
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "robust"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=Path("./results/icl_v"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.lower()
    label_col = args.label_col.lower()
    if label_col not in df.columns:
        raise ValueError(f"No se encontró la columna de etiqueta '{label_col}' en {args.data}")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col])

    # Resolver rutas de columnas
    base_cols_dir = Path("./utils/columns/iclv")
    obs_lt_file = args.obs_lt_cols or base_cols_dir / "obs_lt.txt"
    obs_u_file = args.obs_u_cols or base_cols_dir / "obs_u.txt"
    obs_i_file = args.obs_i_cols or base_cols_dir / "obs_i.txt"

    drop_cols = {label_col}
    obs_lt_cols = resolve_cols(df, str(obs_lt_file) if obs_lt_file else None, fallback_numeric=False, drop_cols=drop_cols)
    obs_u_cols = resolve_cols(df, str(obs_u_file) if obs_u_file else None, fallback_numeric=True, drop_cols=drop_cols)
    obs_i_cols = resolve_cols(df, str(obs_i_file) if obs_i_file else None, fallback_numeric=False, drop_cols=drop_cols)

    train_df, val_df = split_train_val(df, label_col=label_col, val_split=args.val_split, seed=args.seed)

    train_ds, val_ds, preproc_lt, preproc_u = build_datasets(
        train_df,
        val_df,
        obs_lt_cols,
        obs_u_cols,
        obs_i_cols,
        label_col,
        num_choices=args.num_choices,
        scaler=args.scaler,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = DeterministicICLV(
        dim_obs_lt=train_ds.obs_lt.shape[1],
        dim_obs_u=train_ds.obs_u.shape[2],
        n_latent=args.n_latent,
        n_indicators=train_ds.indicators.shape[1],
        n_choices=args.num_choices,
        alpha=args.alpha,
        delta_per_alt=args.delta_per_alt,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        tr_metrics = run_epoch(model, train_loader, device, train=True, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device, train=False)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={tr_metrics['loss']:.4f} acc={tr_metrics['acc']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f}"
        )

    # Hessiano sobre train
    def loss_closure():
        out = []
        for obs_lt, obs_u, indicators, choice in train_loader:
            obs_lt = obs_lt.to(device)
            obs_u = obs_u.to(device)
            indicators = indicators.to(device)
            choice_t = torch.as_tensor(choice, device=device, dtype=torch.long)
            o = model(obs_lt, obs_u, indicators, choice_t)
            out.append(o["loss"])
        return torch.stack(out).mean()

    hess = compute_hessian_stats(model, loss_closure)

    # Métricas adicionales: F1 y pseudo-R2 (McFadden) en train y val
    from sklearn.metrics import f1_score

    def pseudo_r2(ll_model: float, y_true: np.ndarray, num_choices: int) -> float:
        if len(y_true) == 0:
            return float("nan")
        if num_choices == 2:
            p = np.clip(y_true.mean(), 1e-6, 1 - 1e-6)
            ll_null = (y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).sum()
        else:
            counts = np.bincount(y_true.astype(int), minlength=num_choices)
            probs = counts / counts.sum()
            probs = np.clip(probs, 1e-6, 1.0)
            ll_null = np.log(probs[y_true.astype(int)]).sum()
        return 1 - (ll_model / ll_null)

    f1_tr = f1_score(tr_metrics["y_true"], tr_metrics["y_pred"], zero_division=0) if len(tr_metrics["y_true"]) else float("nan")
    f1_val = f1_score(val_metrics["y_true"], val_metrics["y_pred"], zero_division=0) if len(val_metrics["y_true"]) else float("nan")
    r2_tr = pseudo_r2(tr_metrics["log_likelihood"], tr_metrics["y_true"], args.num_choices)
    r2_val = pseudo_r2(val_metrics["log_likelihood"], val_metrics["y_true"], args.num_choices)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.results_dir / "model.pt")
    with open(args.results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_loss": tr_metrics["loss"],
                "train_acc": tr_metrics["acc"],
                "train_f1": f1_tr,
                "train_log_likelihood": tr_metrics["log_likelihood"],
                "train_pseudo_r2": r2_tr,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_f1": f1_val,
                "val_log_likelihood": val_metrics["log_likelihood"],
                "val_pseudo_r2": r2_val,
                "obs_lt_cols": obs_lt_cols,
                "obs_u_cols": obs_u_cols,
                "obs_i_cols": obs_i_cols,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(args.results_dir / "hessian.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "theta": hess.theta.tolist(),
                "std": hess.std.tolist(),
                "tstat": hess.tstat.tolist(),
                "names": hess.names,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    torch.save(preproc_lt, args.results_dir / "preproc_lt.pkl")
    torch.save(preproc_u, args.results_dir / "preproc_u.pkl")
    print(f"Guardado en {args.results_dir}")


if __name__ == "__main__":
    main()
