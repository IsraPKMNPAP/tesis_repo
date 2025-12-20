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
from torch.utils.data import DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler

# Asegurar import relativo desde dataset_bicicletas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loading.icl_v import ICLVDataset
from src.models.icl_v import DeterministicICLV, compute_hessian_stats
from src.data_cleaning.cleaning import categorias_a_str, convertir_a_categorico
from utils.features import load_features_file
from utils.results_io import (
    ensure_dir,
    save_model_pickle,
    save_text,
    compute_run_hash,
    artifact_name,
    register_run,
)


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
    df_prep = convertir_a_categorico(categorias_a_str(df[cols].copy()))
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


def encode_indicator_blocks(
    df_tr: pd.DataFrame, df_val: pd.DataFrame, cols: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Convierte indicadores mixtos a numerico (factoriza strings/categorias)."""
    tr_blocks = []
    val_blocks = []
    for col in cols:
        if pd.api.types.is_numeric_dtype(df_tr[col]):
            tr_col = df_tr[col].fillna(df_tr[col].median())
            val_col = df_val[col].fillna(df_tr[col].median())
        else:
            # Factorizar sobre train y aplicar mapping al val; out-of-vocab -> -1
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


def resolve_cols(
    df: pd.DataFrame,
    base_features_file: str | None,
    explicit_cols: Sequence[str] | None,
    cols_file: str | None,
    drop_cols: set,
) -> List[str]:
    """Resuelve columnas a usar, con prioridad: explicit -> file -> base features file -> infer numeric."""
    if explicit_cols:
        cols = list(explicit_cols)
    elif cols_file:
        cols = load_features_file(cols_file)
    elif base_features_file:
        cols = load_features_file(base_features_file)
    else:
        cols = []
    if not cols:
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
    X_lt_val = preproc_lt.transform(convertir_a_categorico(categorias_a_str(df_val[obs_lt_cols].copy())))

    X_u_tr, preproc_u = prepare_preprocessor(df_tr, obs_u_cols, scaler=scaler)
    X_u_val = preproc_u.transform(convertir_a_categorico(categorias_a_str(df_val[obs_u_cols].copy())))

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

    avg_loss = total_loss / max(1, total)
    avg_choice = total_choice / max(1, total)
    avg_meas = total_meas / max(1, total)
    avg_ll = total_ll / max(1, total)
    acc = correct / max(1, total)
    return {
        "loss": avg_loss,
        "loss_choice": avg_choice,
        "loss_meas": avg_meas,
        "log_likelihood": total_ll,
        "avg_log_likelihood": avg_ll,
        "acc": acc,
    }


def main():
    ap = argparse.ArgumentParser(description="Entrena un ICLV determinista amortizado (sin integracion Monte Carlo).")
    ap.add_argument("--pkl", type=str, default="data/processed/multimodal_av_join_audio_cached.pkl", help="Pickle multimodal de entrada")
    ap.add_argument("--label-col", type=str, default="action_proc")
    ap.add_argument("--features-file", type=str, default="utils/feature_sets/exp1.json", help="Archivo con columnas base")
    ap.add_argument("--obs-lt-cols", nargs="*", default=None, help="Columnas observables para latentes")
    ap.add_argument("--obs-lt-cols-file", type=str, default=None, help="Archivo con columnas OBS_LT (json/txt)")
    ap.add_argument("--obs-u-cols", nargs="*", default=None, help="Columnas observables para utilidad")
    ap.add_argument("--obs-u-cols-file", type=str, default=None, help="Archivo con columnas OBS_U (json/txt)")
    ap.add_argument("--indicator-cols", nargs="*", default=None, help="Indicadores OBS_I para el bloque de medicion")
    ap.add_argument("--indicator-cols-file", type=str, default=None, help="Archivo con columnas OBS_I (json/txt)")
    ap.add_argument("--n-latent", type=int, default=3, help="Numero de variables latentes")
    ap.add_argument("--alpha", type=float, default=1.0, help="Peso de la loss de medicion")
    ap.add_argument("--delta-shared", action="store_true", help="Usar un delta compartido en vez de por alternativa")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tabular-scaler", type=str, default="standard", choices=["standard", "robust"])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        raise FileNotFoundError(f"No existe el pickle {pkl_path}")
    df = pd.read_pickle(pkl_path).reset_index(drop=True)

    # Resolver columnas
    drop_cols = {
        args.label_col,
        "frames_route",
        "audio_cached_path",
        "timestamp",
        "window",
        "participant",
        "session_id",
    }

    obs_lt_cols = resolve_cols(
        df=df,
        base_features_file=args.features_file,
        explicit_cols=args.obs_lt_cols,
        cols_file=args.obs_lt_cols_file,
        drop_cols=drop_cols,
    )
    obs_u_cols = resolve_cols(
        df=df,
        base_features_file=args.features_file,
        explicit_cols=args.obs_u_cols,
        cols_file=args.obs_u_cols_file,
        drop_cols=drop_cols,
    )
    indicator_cols = resolve_cols(
        df=df,
        base_features_file=args.features_file,
        explicit_cols=args.indicator_cols,
        cols_file=args.indicator_cols_file,
        drop_cols=set(),  # para indicadores permitimos non-numeric; se codifican
    )
    if not indicator_cols:
        indicator_cols = []

    if not obs_lt_cols or not obs_u_cols:
        raise ValueError("No se encontraron columnas validas para obs_lt u obs_u.")

    # Mapear etiquetas
    default_class_map = {
        "accelerate": 0,
        "brake": 1,
        "decelerate": 2,
        "maintain speed": 3,
        "wait": 4,
    }
    if df[args.label_col].dtype == object:
        df[args.label_col] = df[args.label_col].map(default_class_map)
    if df[args.label_col].isna().any():
        df = df.dropna(subset=[args.label_col]).reset_index(drop=True)
    df[args.label_col] = df[args.label_col].astype(int)
    num_choices = int(pd.Series(df[args.label_col]).nunique())

    df_tr, df_val = split_train_val(df, label_col=args.label_col, val_split=args.val_split, seed=args.seed)

    train_ds, val_ds, preproc_lt, preproc_u = build_datasets(
        df_tr=df_tr,
        df_val=df_val,
        obs_lt_cols=obs_lt_cols,
        obs_u_cols=obs_u_cols,
        indicator_cols=indicator_cols,
        label_col=args.label_col,
        num_choices=num_choices,
        scaler=args.tabular_scaler,
    )

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = DeterministicICLV(
        dim_obs_lt=train_ds.obs_lt.shape[1],
        dim_obs_u=train_ds.obs_u.shape[2],
        n_latent=args.n_latent,
        n_indicators=train_ds.indicators.shape[1],
        n_choices=num_choices,
        alpha=args.alpha,
        delta_per_alt=not args.delta_shared,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    history = []
    for epoch in range(1, args.epochs + 1):
        tr_metrics = run_epoch(model, train_loader, device=device, train=True, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device=device, train=False)
        history.append({"epoch": epoch, "train": tr_metrics, "val": val_metrics})
        print(
            f"Epoch {epoch:03d} | "
            f"train loss={tr_metrics['loss']:.4f} acc={tr_metrics['acc']:.3f} ll={tr_metrics['avg_log_likelihood']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.3f} ll={val_metrics['avg_log_likelihood']:.4f}"
        )

    # Hessiano y estadisticos (sobre train completo)
    full_train = ICLVDataset(
        obs_lt=train_ds.obs_lt,
        obs_u=train_ds.obs_u,
        indicators=train_ds.indicators,
        choices=train_ds.choices,
        num_choices=num_choices,
    )
    obs_lt_full = full_train.obs_lt.to(device)
    obs_u_full = full_train.obs_u.to(device)
    ind_full = full_train.indicators.to(device)
    choice_full = full_train.choices.to(device)

    def loss_closure():
        out = model(obs_lt_full, obs_u_full, ind_full, choice_full)
        return out["loss"]

    hess_res = compute_hessian_stats(model, loss_closure)

    # Reporte
    results_dir = Path("results")
    ensure_dir(results_dir)
    base_config = {
        "pkl": str(pkl_path),
        "label_col": args.label_col,
        "features_file": args.features_file,
        "obs_lt_cols": list(obs_lt_cols),
        "obs_u_cols": list(obs_u_cols),
        "indicator_cols": list(indicator_cols),
        "obs_lt_cols_file": args.obs_lt_cols_file,
        "obs_u_cols_file": args.obs_u_cols_file,
        "indicator_cols_file": args.indicator_cols_file,
        "n_latent": args.n_latent,
        "alpha": args.alpha,
        "delta_shared": args.delta_shared,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "val_split": args.val_split,
        "tabular_scaler": args.tabular_scaler,
        "seed": args.seed,
        "device": str(device),
    }
    run_hash = compute_run_hash(base_config, sys.argv, model="ICLV")

    report_lines = []
    report_lines.append("=== ICLV determinista ===")
    report_lines.append(
        f"Train acc={history[-1]['train']['acc']:.4f} loglik_sum={history[-1]['train']['log_likelihood']:.4f} "
        f"loglik_mean={history[-1]['train']['avg_log_likelihood']:.4f}"
    )
    report_lines.append(
        f"Val   acc={history[-1]['val']['acc']:.4f} loglik_sum={history[-1]['val']['log_likelihood']:.4f} "
        f"loglik_mean={history[-1]['val']['avg_log_likelihood']:.4f}"
    )
    report_lines.append("")
    report_lines.append("Parametros (theta, se, tstat):")
    for name, theta, se, tstat in zip(hess_res.names, hess_res.theta.tolist(), hess_res.std.tolist(), hess_res.tstat.tolist()):
        report_lines.append(f"{name}: theta={theta:.6f} se={se:.6f} t={tstat:.3f}")
    report_text = "\n".join(report_lines)
    save_text(report_text, results_dir / artifact_name("ICLV", "eval_report", run_hash, "txt"))

    config_path = results_dir / artifact_name("ICLV", "config", run_hash, "json")
    config_path.write_text(json.dumps(base_config, indent=2), encoding="utf-8")

    params_table = pd.DataFrame(
        {
            "name": hess_res.names,
            "theta": hess_res.theta.cpu().numpy(),
            "std_error": hess_res.std.cpu().numpy(),
            "t_stat": hess_res.tstat.cpu().numpy(),
        }
    )
    params_table.to_csv(results_dir / artifact_name("ICLV", "params", run_hash, "csv"), index=False)

    torch.save(model.state_dict(), results_dir / artifact_name("ICLV", "model", run_hash, "pt"))
    save_model_pickle(preproc_lt, results_dir / artifact_name("ICLV", "preproc_lt", run_hash, "pkl"))
    save_model_pickle(preproc_u, results_dir / artifact_name("ICLV", "preproc_u", run_hash, "pkl"))

    # Registrar corrida
    register_run(results_dir, run_hash, "ICLV", cmd=" ".join(sys.argv), config=base_config)

    # Guardar historial simple
    hist_path = results_dir / artifact_name("ICLV", "history", run_hash, "json")
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
