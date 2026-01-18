from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data_loading.icl_v import ICLVDataset
from src.models.icl_v_v2 import ICLV
from utils.features import load_features_file
from utils.metrics import classification_metrics, pseudo_r2_mcfadden, save_metrics
from utils.run_utils import next_run_dir, save_run_metadata
from utils.splits import split_by_subject_train_val_test, save_split_info


def prepare_preprocessor(
    df: pd.DataFrame, cols: Sequence[str], scaler: str = "standard", cat_unique_threshold: int = 4
) -> Tuple[np.ndarray, object]:
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


def to_float_array(mat) -> np.ndarray:
    try:
        arr = mat.toarray()
    except Exception:
        arr = np.asarray(mat)
    return arr.astype(np.float32)


def build_design_matrices(
    df_tr: pd.DataFrame,
    df_val: pd.DataFrame,
    df_te: pd.DataFrame,
    obs_lt_cols: List[str],
    obs_u_cols: List[str],
    obs_i_cols: List[str],
    scaler: str,
    cat_unique_threshold: int,
    min_var: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    X_lt_tr, preproc_lt = prepare_preprocessor(df_tr, obs_lt_cols, scaler=scaler, cat_unique_threshold=cat_unique_threshold)
    X_lt_val = preproc_lt.transform(df_val[obs_lt_cols].copy())
    X_lt_te = preproc_lt.transform(df_te[obs_lt_cols].copy())

    X_u_tr, preproc_u = prepare_preprocessor(df_tr, obs_u_cols, scaler=scaler, cat_unique_threshold=cat_unique_threshold)
    X_u_val = preproc_u.transform(df_val[obs_u_cols].copy())
    X_u_te = preproc_u.transform(df_te[obs_u_cols].copy())
    try:
        feat_names_u = list(preproc_u.get_feature_names_out(obs_u_cols))
    except Exception:
        feat_names_u = list(obs_u_cols)

    X_i_tr, preproc_i = prepare_preprocessor(df_tr, obs_i_cols, scaler=scaler, cat_unique_threshold=cat_unique_threshold) if obs_i_cols else (np.zeros((len(df_tr), 0)), None)
    X_i_val = preproc_i.transform(df_val[obs_i_cols].copy()) if obs_i_cols else np.zeros((len(df_val), 0))
    X_i_te = preproc_i.transform(df_te[obs_i_cols].copy()) if obs_i_cols else np.zeros((len(df_te), 0))

    X_u_tr = to_float_array(X_u_tr)
    X_u_val = to_float_array(X_u_val)
    X_u_te = to_float_array(X_u_te)

    # filter low variance
    if X_u_tr.shape[1] > 0:
        var = np.var(X_u_tr, axis=0)
        mask = var >= min_var
        X_u_tr = X_u_tr[:, mask]
        X_u_val = X_u_val[:, mask]
        X_u_te = X_u_te[:, mask]
        feat_names_u = [n for n, keep in zip(feat_names_u, mask) if keep]

    return (
        to_float_array(X_lt_tr),
        to_float_array(X_lt_val),
        to_float_array(X_lt_te),
        X_u_tr,
        X_u_val,
        X_u_te,
        to_float_array(X_i_tr),
        to_float_array(X_i_val),
        to_float_array(X_i_te),
        feat_names_u,
    )


def run_epoch(model, loader, device, n_draws: int, train: bool, optimizer=None):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_ll = 0.0
    y_true_all = []
    y_prob_all = []
    total = 0
    with torch.set_grad_enabled(train):
        for obs_lt, obs_u, indicators, choice in loader:
            obs_lt = obs_lt.to(device)
            obs_u = obs_u.to(device)
            indicators = indicators.to(device)
            choice_t = choice.to(device)
            out = model(obs_lt, obs_u, indicators, choice_t, n_draws=n_draws)
            loss = out["loss"]
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * obs_lt.size(0)
            total_ll += float(out["log_likelihood"].item())
            prob1 = out["probs"][:, 1] if out["probs"].shape[1] > 1 else out["probs"][:, 0]
            y_true_all.append(choice_t.detach().cpu().numpy())
            y_prob_all.append(prob1.detach().cpu().numpy())
            total += obs_lt.size(0)
    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.array([])
    return {"loss": total_loss / max(1, total), "log_likelihood": total_ll, "n": total, "y_true": y_true, "y_prob": y_prob}


def main() -> None:
    parser = argparse.ArgumentParser(description="ICLV v2 (SML) minimal.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--obs-lt-cols", type=Path, required=True)
    parser.add_argument("--obs-u-cols", type=Path, required=True)
    parser.add_argument("--obs-i-cols", type=Path, required=True)
    parser.add_argument("--label-col", type=str, default="bought")
    parser.add_argument("--num-choices", type=int, default=2)
    parser.add_argument("--n-latent", type=int, default=1)
    parser.add_argument("--beta-per-alt", action="store_true")
    parser.add_argument("--delta-per-alt", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-draws", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--cat-unique-threshold", type=int, default=4)
    parser.add_argument("--min-var", type=float, default=1e-6)
    parser.add_argument("--obs-u-buy-only", action="store_true", help="Aplica obs_u solo a alternativa buy (alt=1).")
    parser.add_argument("--results-dir", type=Path, default=Path("./results/icl_v_v2"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.lower()
    label_col = args.label_col.lower()
    if label_col not in df.columns:
        raise ValueError(f"No se encontro etiqueta '{label_col}'.")
    if "subject" not in df.columns and "id_sub" in df.columns:
        df["subject"] = df["id_sub"].astype(str)
    if "subject" not in df.columns:
        raise ValueError("Se requiere columna 'subject' para split por sujeto.")

    obs_lt_cols = [c.strip().lower() for c in load_features_file(args.obs_lt_cols)]
    obs_u_cols = [c.strip().lower() for c in load_features_file(args.obs_u_cols)]
    obs_i_cols = [c.strip().lower() for c in load_features_file(args.obs_i_cols)]
    obs_lt_cols = [c for c in obs_lt_cols if c in df.columns]
    obs_u_cols = [c for c in obs_u_cols if c in df.columns]
    obs_i_cols = [c for c in obs_i_cols if c in df.columns]
    if not obs_u_cols:
        raise ValueError("obs_u_cols quedo vacio.")

    train_df, val_df, test_df, split_info = split_by_subject_train_val_test(
        df, subject_col="subject", val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )

    (
        X_lt_tr,
        X_lt_val,
        X_lt_te,
        X_u_tr,
        X_u_val,
        X_u_te,
        X_i_tr,
        X_i_val,
        X_i_te,
        _,
    ) = build_design_matrices(
        train_df,
        val_df,
        test_df,
        obs_lt_cols,
        obs_u_cols,
        obs_i_cols,
        scaler="standard",
        cat_unique_threshold=args.cat_unique_threshold,
        min_var=args.min_var,
    )

    if args.obs_u_buy_only:
        if args.num_choices < 2:
            raise ValueError("obs_u_buy_only requiere num_choices >= 2.")
        def expand_buy_only(x: np.ndarray) -> np.ndarray:
            out = np.zeros((x.shape[0], args.num_choices, x.shape[1]), dtype=np.float32)
            out[:, 1, :] = x
            return out
        X_u_tr = expand_buy_only(X_u_tr)
        X_u_val = expand_buy_only(X_u_val)
        X_u_te = expand_buy_only(X_u_te)

    y_tr = pd.to_numeric(train_df[label_col], errors="coerce").to_numpy(dtype=int)
    y_val = pd.to_numeric(val_df[label_col], errors="coerce").to_numpy(dtype=int)
    y_te = pd.to_numeric(test_df[label_col], errors="coerce").to_numpy(dtype=int)

    train_ds = ICLVDataset(X_lt_tr, X_u_tr, X_i_tr, y_tr, args.num_choices)
    val_ds = ICLVDataset(X_lt_val, X_u_val, X_i_val, y_val, args.num_choices)
    test_ds = ICLVDataset(X_lt_te, X_u_te, X_i_te, y_te, args.num_choices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ICLV(
        dim_obs_lt=X_lt_tr.shape[1],
        dim_obs_u=X_u_tr.shape[1],
        n_latent=args.n_latent,
        n_indicators=X_i_tr.shape[1],
        n_choices=args.num_choices,
        alpha=args.alpha,
        delta_per_alt=args.delta_per_alt,
        beta_per_alt=args.beta_per_alt,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, args.n_draws, train=True, optimizer=optim)
        val = run_epoch(model, val_loader, device, args.n_draws, train=False)
        tr_cls = classification_metrics(tr["y_true"], tr["y_prob"])
        val_cls = classification_metrics(val["y_true"], val["y_prob"])
        print(
            f"Epoch {epoch}/{args.epochs} | tr_loss={tr['loss']:.4f} tr_acc={tr_cls['acc']:.3f} tr_f1={tr_cls['f1_macro']:.3f} "
            f"val_loss={val['loss']:.4f} val_acc={val_cls['acc']:.3f} val_f1={val_cls['f1_macro']:.3f}"
        )

    tr = run_epoch(model, train_loader, device, args.n_draws, train=False)
    val = run_epoch(model, val_loader, device, args.n_draws, train=False)
    te = run_epoch(model, test_loader, device, args.n_draws, train=False)

    def pack_metrics(m):
        cls = classification_metrics(m["y_true"], m["y_prob"])
        nll = -m["log_likelihood"] / max(1, m["n"])
        k = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n = max(1, m["n"])
        y = np.asarray(m["y_true"])
        p = y.mean() if y.size else 0.5
        p = min(max(p, 1e-6), 1 - 1e-6)
        ll_null = (y * np.log(p) + (1 - y) * np.log(1 - p)).sum() if y.size else float("nan")
        aic = 2 * k - 2 * m["log_likelihood"]
        bic = np.log(n) * k - 2 * m["log_likelihood"]
        llr = 2 * (m["log_likelihood"] - ll_null) if y.size else float("nan")
        return {
            **cls,
            "mean_nll": float(nll),
            "log_likelihood": float(m["log_likelihood"]),
            "aic": float(aic),
            "bic": float(bic),
            "loglik_null": float(ll_null),
            "loglik_ratio": float(llr),
            "pseudo_r2": float(pseudo_r2_mcfadden(m["log_likelihood"], m["y_true"], args.num_choices)),
        }

    run_dir = next_run_dir(args.results_dir)
    torch.save(model.state_dict(), run_dir / "model_last.pt")
    save_split_info(split_info, run_dir)
    save_run_metadata(args, run_dir)
    metrics = {
        "train": pack_metrics(tr),
        "val": pack_metrics(val),
        "test": pack_metrics(te),
        "obs_lt_cols": obs_lt_cols,
        "obs_u_cols": obs_u_cols,
        "obs_i_cols": obs_i_cols,
    }
    save_metrics(metrics, run_dir)
    print(f"Guardado en {run_dir}")


if __name__ == "__main__":
    main()
