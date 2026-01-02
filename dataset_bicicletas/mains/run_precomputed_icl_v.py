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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loading.precomputed_icl_v import PrecomputedICLVDataset, collate_precomputed_icl_v
from src.models.icl_v import PrecomputedEmbeddingICLV
from utils.features import load_features_file
from utils.results_io import (
    ensure_dir,
    save_model_pickle,
    save_text,
    compute_run_hash,
    artifact_name,
    register_run,
)
from src.data_cleaning.cleaning import categorias_a_str, convertir_a_categorico


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
    return df.iloc[~mask].reset_index(drop=True), df.iloc[mask].reset_index(drop=True)


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


def resolve_cols(df: pd.DataFrame, explicit: Sequence[str] | None, file_path: str | None, drop_cols: set) -> List[str]:
    if explicit:
        cols = list(explicit)
    elif file_path:
        cols = load_features_file(file_path)
    else:
        cols = []
    if not cols:
        cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if c in df.columns]
    return cols


def main():
    ap = argparse.ArgumentParser(description="ICLV con embeddings precalculados (CLIP/wav2vec) congelados.")
    ap.add_argument("--pkl", type=str, default="data/processed/multimodal_av_join_audio_cached.pkl")
    ap.add_argument("--label-col", type=str, default="action_proc")
    ap.add_argument("--obs-lt-cols-file", type=str, default=None)
    ap.add_argument("--obs-u-cols-file", type=str, default=None)
    ap.add_argument("--indicator-cols-file", type=str, default=None)
    ap.add_argument("--obs-lt-cols", nargs="*", default=None)
    ap.add_argument("--obs-u-cols", nargs="*", default=None)
    ap.add_argument("--indicator-cols", nargs="*", default=None)
    ap.add_argument("--features-file", type=str, default="utils/feature_sets/exp1.json", help="Fallback si no se dan listas")
    ap.add_argument("--vid-emb-col", type=str, default="clip_emb", help="Columna con embedding de video precalculado (.pt/.npy)")
    ap.add_argument("--aud-emb-col", type=str, default="wav2vec_emb", help="Columna con embedding de audio precalculado (.pt/.npy)")
    ap.add_argument("--vid-emb-dim", type=int, default=512)
    ap.add_argument("--aud-emb-dim", type=int, default=512)
    ap.add_argument("--n-latent", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--delta-shared", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
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

    drop_cols = {args.label_col, args.vid_emb_col, args.aud_emb_col}
    obs_lt_cols = resolve_cols(df, args.obs_lt_cols, args.obs_lt_cols_file or args.features_file, drop_cols)
    obs_u_cols = resolve_cols(df, args.obs_u_cols, args.obs_u_cols_file or args.features_file, drop_cols)
    indicator_cols = resolve_cols(df, args.indicator_cols, args.indicator_cols_file, set())
    if not indicator_cols:
        indicator_cols = []

    # Label mapping
    default_class_map = {
        "accelerate": 0,
        "brake": 1,
        "decelerate": 2,
        "maintain speed": 3,
        "wait": 4,
    }
    if df[args.label_col].dtype == object:
        df[args.label_col] = df[args.label_col].map(default_class_map)
    df = df.dropna(subset=[args.label_col]).reset_index(drop=True)
    df[args.label_col] = df[args.label_col].astype(int)
    num_choices = int(pd.Series(df[args.label_col]).nunique())

    df_tr, df_val = split_train_val(df, label_col=args.label_col, val_split=args.val_split, seed=args.seed)

    # OBS_LT preprocessing
    X_lt_tr_mat, preproc_lt = prepare_preprocessor(df_tr, obs_lt_cols, scaler=args.tabular_scaler)
    X_lt_val_mat = preproc_lt.transform(convertir_a_categorico(categorias_a_str(df_val[obs_lt_cols].copy())))
    # OBS_U preprocessing
    X_u_tr_mat, preproc_u = prepare_preprocessor(df_tr, obs_u_cols, scaler=args.tabular_scaler)
    X_u_val_mat = preproc_u.transform(convertir_a_categorico(categorias_a_str(df_val[obs_u_cols].copy())))
    # Indicadores
    ind_tr_mat, ind_val_mat = encode_indicator_blocks(df_tr[indicator_cols].copy(), df_val[indicator_cols].copy(), indicator_cols)
    y_tr = pd.to_numeric(df_tr[args.label_col], errors="coerce").to_numpy(dtype=np.int64)
    y_val = pd.to_numeric(df_val[args.label_col], errors="coerce").to_numpy(dtype=np.int64)

    train_ds = PrecomputedICLVDataset(
        df=df_tr,
        obs_lt_array=to_float_array(X_lt_tr_mat),
        obs_u_array=to_float_array(X_u_tr_mat),
        indicator_array=ind_tr_mat,
        choice_array=y_tr,
        vid_emb_col=args.vid_emb_col,
        aud_emb_col=args.aud_emb_col,
    )
    val_ds = PrecomputedICLVDataset(
        df=df_val,
        obs_lt_array=to_float_array(X_lt_val_mat),
        obs_u_array=to_float_array(X_u_val_mat),
        indicator_array=ind_val_mat,
        choice_array=y_val,
        vid_emb_col=args.vid_emb_col,
        aud_emb_col=args.aud_emb_col,
    )

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = PrecomputedEmbeddingICLV(
        tab_in_dim=train_ds.obs_lt.shape[1],
        dim_obs_u=train_ds.obs_u.shape[-1],
        n_indicators=train_ds.indicators.shape[1],
        n_choices=num_choices,
        emb_vid_dim=args.vid_emb_dim,
        emb_aud_dim=args.aud_emb_dim,
        shared_dim=args.n_latent,
        alpha=args.alpha,
        delta_per_alt=not args.delta_shared,
        fuse_dropout=0.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_precomputed_icl_v)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_precomputed_icl_v)

    def run_epoch(loader, train_flag: bool):
        if train_flag:
            model.train()
        else:
            model.eval()
        total_loss = total_choice = total_meas = total_ll = 0.0
        correct = total = 0
        for batch in loader:
            x_tab = batch.x_tab.to(device)
            vid_emb = batch.vid_emb.to(device) if batch.vid_emb is not None else None
            aud_emb = batch.aud_emb.to(device) if batch.aud_emb is not None else None
            obs_u = batch.obs_u.to(device)
            indicators = batch.indicators.to(device)
            y = batch.y.to(device)

            out = model(x_tab, vid_emb, aud_emb, obs_u, indicators, y)
            loss = out["loss"]
            if train_flag:
                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

            bsz = x_tab.size(0)
            total_loss += float(loss.item()) * bsz
            total_choice += float(out["loss_choice"].item()) * bsz
            total_meas += float(out["loss_meas"].item()) * bsz
            total_ll += float(out["log_likelihood"].item())
            preds = out["logp"].argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += bsz

        return {
            "loss": total_loss / max(1, total),
            "loss_choice": total_choice / max(1, total),
            "loss_meas": total_meas / max(1, total),
            "log_likelihood": total_ll,
            "avg_log_likelihood": total_ll / max(1, total),
            "acc": correct / max(1, total),
        }

    history = []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(train_loader, True)
        val = run_epoch(val_loader, False)
        history.append({"epoch": epoch, "train": tr, "val": val})
        print(
            f"Epoch {epoch:03d} | train loss={tr['loss']:.4f} acc={tr['acc']:.3f} ll={tr['avg_log_likelihood']:.4f} | "
            f"val loss={val['loss']:.4f} acc={val['acc']:.3f} ll={val['avg_log_likelihood']:.4f}"
        )

    results_dir = Path("results")
    ensure_dir(results_dir)
    base_config = {
        "pkl": str(pkl_path),
        "label_col": args.label_col,
        "obs_lt_cols": list(obs_lt_cols),
        "obs_u_cols": list(obs_u_cols),
        "indicator_cols": list(indicator_cols),
        "obs_lt_cols_file": args.obs_lt_cols_file,
        "obs_u_cols_file": args.obs_u_cols_file,
        "indicator_cols_file": args.indicator_cols_file,
        "vid_emb_col": args.vid_emb_col,
        "aud_emb_col": args.aud_emb_col,
        "vid_emb_dim": args.vid_emb_dim,
        "aud_emb_dim": args.aud_emb_dim,
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
        "grad_clip": args.grad_clip,
    }
    run_hash = compute_run_hash(base_config, sys.argv, model="PRECOMP_ICLV")

    report = [
        "=== ICLV con embeddings precalculados ===",
        f"Train acc={history[-1]['train']['acc']:.4f} loglik_sum={history[-1]['train']['log_likelihood']:.4f} "
        f"loglik_mean={history[-1]['train']['avg_log_likelihood']:.4f}",
        f"Val   acc={history[-1]['val']['acc']:.4f} loglik_sum={history[-1]['val']['log_likelihood']:.4f} "
        f"loglik_mean={history[-1]['val']['avg_log_likelihood']:.4f}",
    ]
    save_text("\n".join(report), results_dir / artifact_name("PRECOMP_ICLV", "eval_report", run_hash, "txt"))
    (results_dir / artifact_name("PRECOMP_ICLV", "config", run_hash, "json")).write_text(
        json.dumps(base_config, indent=2), encoding="utf-8"
    )
    torch.save(model.state_dict(), results_dir / artifact_name("PRECOMP_ICLV", "model", run_hash, "pt"))
    save_model_pickle(preproc_lt, results_dir / artifact_name("PRECOMP_ICLV", "preproc_lt", run_hash, "pkl"))
    save_model_pickle(preproc_u, results_dir / artifact_name("PRECOMP_ICLV", "preproc_u", run_hash, "pkl"))
    hist_path = results_dir / artifact_name("PRECOMP_ICLV", "history", run_hash, "json")
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    register_run(results_dir, run_hash, "PRECOMP_ICLV", cmd=" ".join(sys.argv), config=base_config)


if __name__ == "__main__":
    main()
