from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from torch.utils.data import DataLoader
try:
    from torch.nn.utils import parameters_to_vector, vector_to_parameters
except Exception:
    from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading.icl_v import ICLVDataset
from src.models.icl_v import DeterministicICLV, compute_hessian_stats
from utils.features import load_features_file
from utils.metrics import classification_metrics, pseudo_r2_mcfadden, save_metrics, summarize_coefs
from utils.run_utils import next_run_dir, save_run_metadata
from utils.splits import split_by_subject_train_val_test, save_split_info


def to_float_array(mat) -> np.ndarray:
    try:
        arr = mat.toarray()
    except Exception:
        arr = np.asarray(mat)
    return arr.astype(np.float32)


def prepare_preprocessor(df: pd.DataFrame, cols: Sequence[str], scaler: str = "standard", cat_unique_threshold: int = 50):
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
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    mat = preprocessor.fit_transform(df_prep)
    return mat, preprocessor


def encode_indicator_blocks(df_tr: pd.DataFrame, df_val: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
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
    cat_unique_threshold: int = 50,
):
    X_lt_tr, preproc_lt = prepare_preprocessor(df_tr, obs_lt_cols, scaler=scaler, cat_unique_threshold=cat_unique_threshold)
    X_lt_val = preproc_lt.transform(df_val[obs_lt_cols].copy())

    X_u_tr, preproc_u = prepare_preprocessor(df_tr, obs_u_cols, scaler=scaler, cat_unique_threshold=cat_unique_threshold)
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
    y_true_all = []
    y_prob_all = []
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
        prob1 = torch.exp(out["logp"][:, 1]) if out["logp"].shape[1] > 1 else torch.exp(out["logp"][:, 0])
        y_true_all.append(choice_t.detach().cpu().numpy())
        y_prob_all.append(prob1.detach().cpu().numpy())
        total += obs_lt.size(0)

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.array([])
    avg_loss = total_loss / max(1, total)
    avg_choice = total_choice / max(1, total)
    avg_meas = total_meas / max(1, total)
    return {
        "loss": avg_loss,
        "loss_choice": avg_choice,
        "loss_meas": avg_meas,
        "log_likelihood": total_ll,
        "n": total,
        "y_true": y_true,
        "y_prob": y_prob,
    }


def main():
    parser = argparse.ArgumentParser(description="ICLV determinista clasico para NEUMA (split por sujeto).")
    parser.add_argument("--data", type=Path, default=Path("./data/processed/multimodal_join.csv"))
    parser.add_argument("--label-col", type=str, default="bought")
    parser.add_argument("--obs-lt-cols", type=str, default=None)
    parser.add_argument("--obs-u-cols", type=str, default=None)
    parser.add_argument("--obs-i-cols", type=str, default=None)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-latent", type=int, default=3)
    parser.add_argument("--delta-per-alt", action="store_true")
    parser.add_argument("--beta-per-alt", action="store_true", help="Usar betas distintos por alternativa.")
    parser.add_argument("--num-choices", type=int, default=2)
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "robust"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=Path("./results/icl_v"))
    parser.add_argument("--debug", action="store_true", help="Imprime diagnosticos por epoca.")
    parser.add_argument("--hessian-choice-only", action="store_true", help="Hessiano solo de loss_choice.")
    parser.add_argument("--cat-unique-threshold", type=int, default=50, help="Nunique<=threshold -> categorico + one-hot.")
    parser.add_argument("--hessian-beta-only", action="store_true", help="Hessiano solo para betas de utilidad.")
    parser.add_argument("--hessian-double", action="store_true", help="Hessiano en float64 (CPU recomendado).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.lower()
    label_col = args.label_col.lower()
    if label_col not in df.columns:
        raise ValueError(f"No se encontro columna de etiqueta '{label_col}'.")
    if "subject" not in df.columns and "id_sub" in df.columns:
        df["subject"] = df["id_sub"].astype(str)
    if "subject" not in df.columns:
        raise ValueError("Se requiere columna 'subject' para split por sujeto.")

    base_cols_dir = Path("./utils/columns/iclv")
    obs_lt_file = args.obs_lt_cols or base_cols_dir / "obs_lt.txt"
    obs_u_file = args.obs_u_cols or base_cols_dir / "obs_u.txt"
    obs_i_file = args.obs_i_cols or base_cols_dir / "obs_i.txt"

    drop_cols = {label_col}
    obs_lt_raw = [c.strip().lower() for c in load_features_file(obs_lt_file)] if obs_lt_file else []
    obs_u_raw = [c.strip().lower() for c in load_features_file(obs_u_file)] if obs_u_file else []
    obs_i_raw = [c.strip().lower() for c in load_features_file(obs_i_file)] if obs_i_file else []

    obs_lt_cols = resolve_cols(df, str(obs_lt_file) if obs_lt_file else None, fallback_numeric=False, drop_cols=drop_cols)
    obs_u_cols = resolve_cols(df, str(obs_u_file) if obs_u_file else None, fallback_numeric=True, drop_cols=drop_cols)
    obs_i_cols = resolve_cols(df, str(obs_i_file) if obs_i_file else None, fallback_numeric=False, drop_cols=drop_cols)

    missing_lt = [c for c in obs_lt_raw if c not in df.columns]
    missing_u = [c for c in obs_u_raw if c not in df.columns]
    missing_i = [c for c in obs_i_raw if c not in df.columns]
    if missing_lt:
        print(f"[warn] obs_lt missing cols: {missing_lt}")
    if missing_u:
        print(f"[warn] obs_u missing cols: {missing_u}")
    if missing_i:
        print(f"[warn] obs_i missing cols: {missing_i}")

    train_df, val_df, test_df, split_info = split_by_subject_train_val_test(
        df, subject_col="subject", val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    print(
        f"[split] subjects={split_info['n_subjects']} train={split_info['n_train_subjects']} "
        f"val={split_info['n_val_subjects']} test={split_info['n_test_subjects']} | "
        f"rows train={split_info['train_rows']} val={split_info['val_rows']} test={split_info['test_rows']}"
    )

    train_ds, val_ds, preproc_lt, preproc_u = build_datasets(
        train_df,
        val_df,
        obs_lt_cols,
        obs_u_cols,
        obs_i_cols,
        label_col,
        num_choices=args.num_choices,
        scaler=args.scaler,
        cat_unique_threshold=args.cat_unique_threshold,
    )
    # test usando preprocesadores de train
    X_lt_te = preproc_lt.transform(test_df[obs_lt_cols].copy())
    X_u_te = preproc_u.transform(test_df[obs_u_cols].copy())
    if obs_i_cols:
        ind_tr_mat, ind_te_mat = encode_indicator_blocks(train_df[obs_i_cols].copy(), test_df[obs_i_cols].copy(), obs_i_cols)
    else:
        ind_te_mat = np.zeros((len(test_df), 0), dtype=np.float32)
    y_te = pd.to_numeric(test_df[label_col], errors="coerce").to_numpy(dtype=np.int64)
    test_ds = ICLVDataset(
        obs_lt=to_float_array(X_lt_te),
        obs_u=to_float_array(X_u_te),
        indicators=ind_te_mat,
        choices=y_te,
        num_choices=args.num_choices,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = DeterministicICLV(
        dim_obs_lt=train_ds.obs_lt.shape[1],
        dim_obs_u=train_ds.obs_u.shape[2],
        n_latent=args.n_latent,
        n_indicators=train_ds.indicators.shape[1],
        n_choices=args.num_choices,
        alpha=args.alpha,
        delta_per_alt=args.delta_per_alt,
        beta_per_alt=args.beta_per_alt,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        tr_metrics = run_epoch(model, train_loader, device, train=True, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device, train=False)
        tr_cls = classification_metrics(tr_metrics["y_true"], tr_metrics["y_prob"])
        val_cls = classification_metrics(val_metrics["y_true"], val_metrics["y_prob"])
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={tr_metrics['loss']:.4f} acc={tr_cls['acc']:.3f} f1={tr_cls['f1_macro']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_cls['acc']:.3f} val_f1={val_cls['f1_macro']:.3f}"
        )
        if args.debug:
            obs_u_std = float(train_ds.obs_u.std().item()) if train_ds.obs_u.numel() else float("nan")
            if hasattr(model.beta, "weight"):
                beta_norm = float(model.beta.weight.norm().item())
            else:
                beta_norm = float(model.beta.norm().item())
            delta_norm = float(model.delta.norm().item())
            print(
                f"[debug] loss_choice={tr_metrics['loss_choice']:.4f} loss_meas={tr_metrics['loss_meas']:.4f} "
                f"obs_u_std={obs_u_std:.4f} beta_norm={beta_norm:.4f} delta_norm={delta_norm:.4f}"
            )

    tr_metrics = run_epoch(model, train_loader, device, train=False)
    val_metrics = run_epoch(model, val_loader, device, train=False)
    te_metrics = run_epoch(model, test_loader, device, train=False)

    # Hessian over train for coef stats
    def loss_closure():
        out = []
        for obs_lt, obs_u, indicators, choice in train_loader:
            obs_lt = obs_lt.to(device)
            obs_u = obs_u.to(device)
            indicators = indicators.to(device)
            choice_t = torch.as_tensor(choice, device=device, dtype=torch.long)
            o = model(obs_lt, obs_u, indicators, choice_t)
            out.append(o["loss_choice"] if args.hessian_choice_only else o["loss"])
        return torch.stack(out).mean()

    if args.hessian_beta_only:
        beta_param = model.beta if isinstance(model.beta, torch.nn.Parameter) else model.beta.weight
        params = [beta_param]
        flat_init = parameters_to_vector(params).detach()

        def _wrapped_loss(flat_params: torch.Tensor) -> torch.Tensor:
            vector_to_parameters(flat_params, params)
            return loss_closure()

        if args.hessian_double:
            flat_init = flat_init.double()
            beta_param.data = beta_param.data.double()
        H = torch.autograd.functional.hessian(_wrapped_loss, flat_init)
        vector_to_parameters(flat_init, params)
        eye = torch.eye(H.shape[0], device=H.device, dtype=H.dtype) * 1e-4
        H_safe = H + eye
        H_inv = torch.linalg.pinv(H_safe)
        var = torch.diag(H_inv)
        std = torch.sqrt(torch.clamp(var, min=1e-12))
        theta = flat_init
        hess = type("HessWrap", (), {})()
        hess.theta = theta.detach()
        hess.std = std.detach()
        hess.tstat = (theta / std).detach()
        hess.hessian = H.detach()
        hess.var_covar = H_inv.detach()
        if beta_param.dim() == 2:
            hess.names = [f"beta[{i},{j}]" for i in range(beta_param.shape[0]) for j in range(beta_param.shape[1])]
        else:
            hess.names = [f"beta[{j}]" for j in range(beta_param.shape[0])]
    else:
        hess = compute_hessian_stats(model, loss_closure)

    run_dir = next_run_dir(args.results_dir)
    torch.save(model.state_dict(), run_dir / "model.pt")
    save_split_info(split_info, run_dir)
    save_run_metadata(args, run_dir)

    def pack_iclv_metrics(m):
        cls = classification_metrics(m["y_true"], m["y_prob"])
        nll = -m["log_likelihood"] / max(1, m["n"])
        return {
            **cls,
            "mean_nll": float(nll),
            "log_likelihood": float(m["log_likelihood"]),
            "pseudo_r2": float(pseudo_r2_mcfadden(m["log_likelihood"], m["y_true"], args.num_choices)),
        }

    # utility beta stats (solo betas)
    beta_stats = []
    beta_idx = [i for i, n in enumerate(hess.names) if n.startswith("beta")]
    try:
        feat_names = list(preproc_u.get_feature_names_out(obs_u_cols))
    except Exception:
        feat_names = obs_u_cols
    for alt in range(args.num_choices):
        for j, feat in enumerate(feat_names):
            idx = alt * len(feat_names) + j
            flat_idx = beta_idx[idx] if idx < len(beta_idx) else None
            coef = hess.theta[flat_idx] if flat_idx is not None else np.nan
            sd = hess.std[flat_idx] if flat_idx is not None else np.nan
            tstat = coef / sd if sd == sd and sd != 0 else np.nan
            if tstat == tstat:
                if abs(tstat) >= 2.58:
                    stars = "***"
                elif abs(tstat) >= 1.96:
                    stars = "**"
                elif abs(tstat) >= 1.64:
                    stars = "*"
                else:
                    stars = ""
            else:
                stars = ""
            beta_stats.append(
                {
                    "alt": alt,
                    "feature": feat,
                    "coef": float(coef),
                    "std": float(sd) if sd == sd else np.nan,
                    "tstat": float(tstat) if tstat == tstat else np.nan,
                    "stars": stars,
                }
            )

    metrics = {
        "train": pack_iclv_metrics(tr_metrics),
        "val": pack_iclv_metrics(val_metrics),
        "test": pack_iclv_metrics(te_metrics),
        "obs_lt_cols": obs_lt_cols,
        "obs_u_cols": obs_u_cols,
        "obs_i_cols": obs_i_cols,
        "coef_summary": summarize_coefs(hess.names, hess.theta, hess.std, top_k=5),
        "utility_beta_stats": beta_stats,
    }
    save_metrics(metrics, run_dir)

    # Reemplazar nombres beta por nombres de features (utilidad)
    hess_names = list(hess.names)
    try:
        feat_names = list(preproc_u.get_feature_names_out(obs_u_cols))
    except Exception:
        feat_names = obs_u_cols
    beta_idx = [i for i, n in enumerate(hess_names) if n.startswith("beta")]
    for alt in range(args.num_choices):
        for j, feat in enumerate(feat_names):
            idx = alt * len(feat_names) + j
            if idx < len(beta_idx):
                hess_names[beta_idx[idx]] = f"beta[{alt}].{feat}"
    with open(run_dir / "hessian.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "theta": hess.theta.tolist(),
                "std": hess.std.tolist(),
                "tstat": hess.tstat.tolist(),
                "names": hess_names,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    torch.save(preproc_lt, run_dir / "preproc_lt.pkl")
    torch.save(preproc_u, run_dir / "preproc_u.pkl")
    print(f"Guardado en {run_dir}")


if __name__ == "__main__":
    main()
