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
from src.models.icl_v import (
    DeterministicICLV,
    compute_biogeme_hessian_stats_full,
    compute_choice_hessian_stats_only_utility,
)
from utils.features import load_features_file
from utils.metrics import classification_metrics, pseudo_r2_mcfadden, save_metrics, summarize_coefs
from utils.run_utils import next_run_dir, save_run_metadata
from utils.splits import split_by_subject_train_val_test, save_split_info


class SimpleMNL(torch.nn.Module):
    """Logit multinomial simple (sin latentes ni indicadores)."""

    def __init__(self, dim_obs_u: int, n_choices: int, beta_per_alt: bool = False):
        super().__init__()
        self.n_choices = int(n_choices)
        self.beta_per_alt = bool(beta_per_alt)
        if self.beta_per_alt:
            self.beta = torch.nn.Parameter(torch.zeros(n_choices, dim_obs_u))
        else:
            self.beta = torch.nn.Linear(dim_obs_u, 1, bias=False)
        self.ASC = torch.nn.Parameter(torch.zeros(n_choices))
        self._reset_parameters()

    def _reset_parameters(self):
        if isinstance(self.beta, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(self.beta.weight)
        else:
            torch.nn.init.xavier_uniform_(self.beta)
        torch.nn.init.zeros_(self.ASC)

    def compute_utilities(self, obs_u: torch.Tensor) -> torch.Tensor:
        if obs_u.dim() != 3:
            raise ValueError(f"Se espera obs_u con shape [B, J, dim_obs_u]; se recibio {obs_u.shape}")
        if self.beta_per_alt:
            beta_term = (obs_u * self.beta.unsqueeze(0)).sum(-1)
        else:
            beta_term = self.beta(obs_u).squeeze(-1)
        return beta_term + self.ASC.unsqueeze(0)

    def forward(self, obs_lt, obs_u, indicators, choice):
        V = self.compute_utilities(obs_u)
        logp = torch.nn.functional.log_softmax(V, dim=1)
        loss_choice = torch.nn.functional.nll_loss(logp, choice, reduction="mean")
        ll = logp.gather(1, choice.view(-1, 1)).sum()
        return {
            "loss": loss_choice,
            "logp": logp,
            "LT": None,
            "I_hat": None,
            "loss_choice": loss_choice,
            "loss_meas": torch.tensor(0.0, device=obs_u.device, dtype=loss_choice.dtype),
            "log_likelihood": ll,
            "loglik_choice_sum": ll,
        }


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
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical),
        ]
    )
    mat = preprocessor.fit_transform(df_prep)
    return mat, preprocessor


def filter_low_variance(
    mat_tr,
    mat_val,
    feature_names: Sequence[str],
    min_var: float,
    tag: str,
) -> tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    X_tr = to_float_array(mat_tr)
    X_val = to_float_array(mat_val)
    if X_tr.shape[1] == 0:
        return X_tr, X_val, list(feature_names), np.ones(0, dtype=bool)
    var = np.var(X_tr, axis=0)
    mask = var >= min_var
    dropped = [name for name, keep in zip(feature_names, mask) if not keep]
    if dropped:
        print(f"[prep] dropped {len(dropped)} low-variance {tag} cols (var<{min_var}): {dropped[:20]}")
    kept_names = [name for name, keep in zip(feature_names, mask) if keep]
    return X_tr[:, mask], X_val[:, mask], kept_names, mask


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
    min_var: float = 1e-6,
    obs_u_buy_only: bool = False,
    mnl_only: bool = False,
):
    if mnl_only:
        X_lt_tr = np.zeros((len(df_tr), 0), dtype=np.float32)
        X_lt_val = np.zeros((len(df_val), 0), dtype=np.float32)
        preproc_lt = None
    else:
        X_lt_tr, preproc_lt = prepare_preprocessor(df_tr, obs_lt_cols, scaler=scaler, cat_unique_threshold=cat_unique_threshold)
        X_lt_val = preproc_lt.transform(df_val[obs_lt_cols].copy())

    X_u_tr, preproc_u = prepare_preprocessor(df_tr, obs_u_cols, scaler=scaler, cat_unique_threshold=cat_unique_threshold)
    X_u_val = preproc_u.transform(df_val[obs_u_cols].copy())
    try:
        feat_names_u = list(preproc_u.get_feature_names_out(obs_u_cols))
    except Exception:
        feat_names_u = list(obs_u_cols)
    X_u_tr, X_u_val, feat_names_u, u_mask = filter_low_variance(
        X_u_tr, X_u_val, feat_names_u, min_var=min_var, tag="obs_u"
    )

    if indicator_cols and not mnl_only:
        ind_tr_mat, ind_val_mat = encode_indicator_blocks(df_tr[indicator_cols].copy(), df_val[indicator_cols].copy(), indicator_cols)
    else:
        ind_tr_mat = np.zeros((len(df_tr), 0), dtype=np.float32)
        ind_val_mat = np.zeros((len(df_val), 0), dtype=np.float32)

    y_tr = pd.to_numeric(df_tr[label_col], errors="coerce").to_numpy(dtype=np.int64)
    y_val = pd.to_numeric(df_val[label_col], errors="coerce").to_numpy(dtype=np.int64)

    X_u_tr_final = to_float_array(X_u_tr)
    X_u_val_final = to_float_array(X_u_val)
    if obs_u_buy_only:
        if num_choices < 2:
            raise ValueError("obs_u_buy_only requiere num_choices >= 2.")
        tr_exp = np.zeros((len(X_u_tr_final), num_choices, X_u_tr_final.shape[1]), dtype=np.float32)
        val_exp = np.zeros((len(X_u_val_final), num_choices, X_u_val_final.shape[1]), dtype=np.float32)
        tr_exp[:, 1, :] = X_u_tr_final
        val_exp[:, 1, :] = X_u_val_final
        X_u_tr_final = tr_exp
        X_u_val_final = val_exp

    train_ds = ICLVDataset(
        obs_lt=to_float_array(X_lt_tr),
        obs_u=X_u_tr_final,
        indicators=ind_tr_mat,
        choices=y_tr,
        num_choices=num_choices,
    )
    val_ds = ICLVDataset(
        obs_lt=to_float_array(X_lt_val),
        obs_u=X_u_val_final,
        indicators=ind_val_mat,
        choices=y_val,
        num_choices=num_choices,
    )
    return train_ds, val_ds, preproc_lt, preproc_u, feat_names_u, u_mask


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

        total_loss += float(loss.item())
        total_choice += float(out["loss_choice"].item())
        total_meas += float(out["loss_meas"].item())
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
    # alpha eliminado (loglik conjunta)
    parser.add_argument("--n-latent", type=int, default=3)
    parser.add_argument("--delta-per-alt", action="store_true")
    parser.add_argument("--beta-per-alt", action="store_true", help="Usar betas distintos por alternativa.")
    parser.add_argument("--num-choices", type=int, default=2)
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "robust"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=Path("./results/icl_v"))
    parser.add_argument("--debug", action="store_true", help="Imprime diagnosticos por epoca.")
    parser.add_argument("--cat-unique-threshold", type=int, default=4, help="Nunique<=threshold -> categorico + one-hot.")
    parser.add_argument("--min-var", type=float, default=1e-6, help="Filtro de baja varianza para obs_u.")
    parser.add_argument("--hessian-double", action="store_true", help="Hessiano en float64 (CPU recomendado).")
    parser.add_argument("--obs-u-buy-only", action="store_true", help="Aplica obs_u solo a alternativa buy (alt=1).")
    parser.add_argument("--check-obs-u-identical", action="store_true", help="Diagnostica si obs_u es identico entre alternativas.")
    parser.add_argument("--print-beta-norm", action="store_true", help="Imprime norma de beta por epoca.")
    parser.add_argument("--diag-obs-u", action="store_true", help="Imprime norma/varianza de obs_u por alternativa.")
    parser.add_argument("--mnl-only", action="store_true", help="Usa logit simple (sin latentes ni indicadores).")
    parser.add_argument("--no-indicators", action="store_true", help="Ignora obs_i (indicadores).")
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
    if args.no_indicators:
        obs_i_cols = []

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

    train_ds, val_ds, preproc_lt, preproc_u, feat_names_u, u_mask = build_datasets(
        train_df,
        val_df,
        obs_lt_cols,
        obs_u_cols,
        obs_i_cols,
        label_col,
        num_choices=args.num_choices,
        scaler=args.scaler,
        cat_unique_threshold=args.cat_unique_threshold,
        min_var=args.min_var,
        obs_u_buy_only=args.obs_u_buy_only,
        mnl_only=args.mnl_only,
    )
    # test usando preprocesadores de train
    if args.mnl_only:
        X_lt_te = np.zeros((len(test_df), 0), dtype=np.float32)
    else:
        X_lt_te = preproc_lt.transform(test_df[obs_lt_cols].copy())
    X_u_te = preproc_u.transform(test_df[obs_u_cols].copy())
    if u_mask is not None and len(u_mask):
        X_u_te = to_float_array(X_u_te)[:, u_mask]
    if args.obs_u_buy_only:
        if args.num_choices < 2:
            raise ValueError("obs_u_buy_only requiere num_choices >= 2.")
        X_u_te_2d = to_float_array(X_u_te)
        te_exp = np.zeros((len(X_u_te_2d), args.num_choices, X_u_te_2d.shape[1]), dtype=np.float32)
        te_exp[:, 1, :] = X_u_te_2d
        X_u_te = te_exp
    if obs_i_cols and not args.mnl_only:
        ind_tr_mat, ind_te_mat = encode_indicator_blocks(train_df[obs_i_cols].copy(), test_df[obs_i_cols].copy(), obs_i_cols)
    else:
        ind_te_mat = np.zeros((len(test_df), 0), dtype=np.float32)
    y_te = pd.to_numeric(test_df[label_col], errors="coerce").to_numpy(dtype=np.int64)
    test_ds = ICLVDataset(
        obs_lt=to_float_array(X_lt_te),
        obs_u=X_u_te,
        indicators=ind_te_mat,
        choices=y_te,
        num_choices=args.num_choices,
    )

    if args.check_obs_u_identical:
        obs_u_t = train_ds.obs_u
        if obs_u_t.shape[1] > 1:
            diffs = (obs_u_t - obs_u_t[:, 0:1, :]).abs().max().item()
            if diffs < 1e-8:
                print("[diag] obs_u es identico entre alternativas (todas).")
            else:
                print(f"[diag] max |obs_u - obs_u_alt0| = {diffs:.6f}")
    if args.diag_obs_u:
        obs_u_t = train_ds.obs_u
        if obs_u_t.shape[1] > 1:
            alt0 = obs_u_t[:, 0, :]
            alt1 = obs_u_t[:, 1, :]
            print(
                "[diag] obs_u alt0 mean=%.4f std=%.4f l2=%.4f | alt1 mean=%.4f std=%.4f l2=%.4f"
                % (
                    float(alt0.mean().item()),
                    float(alt0.std().item()),
                    float(alt0.norm().item() / max(1, alt0.shape[0])),
                    float(alt1.mean().item()),
                    float(alt1.std().item()),
                    float(alt1.norm().item() / max(1, alt1.shape[0])),
                )
            )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    if args.mnl_only:
        model = SimpleMNL(
            dim_obs_u=train_ds.obs_u.shape[2],
            n_choices=args.num_choices,
            beta_per_alt=args.beta_per_alt,
        )
    else:
        model = DeterministicICLV(
            dim_obs_lt=train_ds.obs_lt.shape[1],
            dim_obs_u=train_ds.obs_u.shape[2],
            n_latent=args.n_latent,
            n_indicators=train_ds.indicators.shape[1],
            n_choices=args.num_choices,
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
        if args.print_beta_norm and not args.debug:
            if hasattr(model.beta, "weight"):
                beta_norm = float(model.beta.weight.norm().item())
            else:
                beta_norm = float(model.beta.norm().item())
            print(f"[beta] epoch={epoch} beta_norm={beta_norm:.6f}")

    tr_metrics = run_epoch(model, train_loader, device, train=False)
    val_metrics = run_epoch(model, val_loader, device, train=False)
    te_metrics = run_epoch(model, test_loader, device, train=False)

    if args.hessian_double:
        model = model.double()
    batch = {
        "obs_lt": train_ds.obs_lt.to(device, dtype=torch.float64 if args.hessian_double else torch.float32),
        "obs_u": train_ds.obs_u.to(device, dtype=torch.float64 if args.hessian_double else torch.float32),
        "indicators": train_ds.indicators.to(device, dtype=torch.float64 if args.hessian_double else torch.float32),
        "choice": train_ds.choices.to(device),
    }
    hess, biogeme_diag, std_robust_full, t_robust_full = compute_biogeme_hessian_stats_full(model, batch)

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
    beta_idx = [i for i, n in enumerate(hess.names) if "beta" in n]
    feat_names = feat_names_u
    beta_param = model.beta if isinstance(model.beta, torch.nn.Parameter) else model.beta.weight
    beta_flat = beta_param.detach().flatten().cpu().numpy()
    base_alt = 0
    alt_list = [a for a in range(args.num_choices) if a != base_alt]
    for alt_pos, alt in enumerate(alt_list):
        for j, feat in enumerate(feat_names):
            idx = alt_pos * len(feat_names) + j
            flat_idx = beta_idx[idx] if idx < len(beta_idx) else None
            coef = beta_flat[idx] if idx < len(beta_flat) else np.nan
            sd = hess.std[flat_idx] if flat_idx is not None else np.nan
            tstat = hess.tstat[flat_idx] if flat_idx is not None else np.nan
            if flat_idx is not None and flat_idx < len(std_robust_full):
                sd_r = std_robust_full[flat_idx]
                tstat_r = t_robust_full[flat_idx]
            else:
                sd_r = np.nan
                tstat_r = np.nan
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
                    "std_robust": float(sd_r) if sd_r == sd_r else np.nan,
                    "tstat_robust": float(tstat_r) if tstat_r == tstat_r else np.nan,
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
        "biogeme_diag": biogeme_diag,
    }
    save_metrics(metrics, run_dir)

    if hess is not None:
        # Reemplazar nombres beta por nombres de features (utilidad)
        hess_names = list(hess.names)
        beta_idx = [i for i, n in enumerate(hess_names) if n.startswith("beta")]
        base_alt = 0
        alt_list = [a for a in range(args.num_choices) if a != base_alt]
        for alt_pos, alt in enumerate(alt_list):
            for j, feat in enumerate(feat_names):
                idx = alt_pos * len(feat_names) + j
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
