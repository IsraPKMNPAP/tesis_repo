from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading.multimodal_icl_v import MultimodalICLVDataset, collate_fn
from src.models.multimodal_icl_v import MultimodalICLVDeterministic
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


def resolve_cols(df: pd.DataFrame, file_path: str | None, fallback_numeric: bool, drop_cols: set) -> List[str]:
    if file_path:
        cols = [c.strip().lower() for c in load_features_file(file_path)]
    else:
        cols = []
    if not cols and fallback_numeric:
        cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if c in df.columns]
    return cols


def run_epoch(model, loader, device, train=True, optimizer=None, alpha=1.0, pos_weight=None):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_choice = 0.0
    total_meas = 0.0
    total_ll = 0.0
    y_true_all = []
    y_pred_all = []
    total = 0
    with torch.set_grad_enabled(train):
        for obs_lt, obs_u, eeg_emb, img_emb, choice in loader:
            obs_lt = obs_lt.to(device)
            obs_u = obs_u.to(device)
            eeg_emb = eeg_emb.to(device)
            img_emb = img_emb.to(device)
            choice_t = choice.to(device)

            out = model(obs_lt, obs_u, eeg_emb, img_emb, choice_t)
            loss = out["loss"]
            if pos_weight is not None:
                # Reemplazar pérdida de elección por BCE ponderado sobre logits de choice (binario)
                # En binario, usamos logp para prob. de clase 1
                if out["logp"].shape[1] == 2:
                    prob1 = torch.exp(out["logp"][:, 1])
                    choice_float = choice_t.float()
                    bce = torch.nn.functional.binary_cross_entropy(prob1, choice_float, weight=pos_weight.expand_as(choice_float))
                    loss = bce + alpha * out["loss_meas"]
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * obs_lt.size(0)
            total_choice += float(out["loss_choice"].item()) * obs_lt.size(0)
            total_meas += float(out["loss_meas"].item()) * obs_lt.size(0)
            total_ll += float(out["log_likelihood"].item())
            preds = out["logp"].argmax(dim=1)
            y_true_all.append(choice_t.cpu())
            y_pred_all.append(preds.cpu())
            total += obs_lt.size(0)

    y_true = torch.cat(y_true_all).numpy() if y_true_all else np.array([])
    y_pred = torch.cat(y_pred_all).numpy() if y_pred_all else np.array([])
    acc = accuracy_score(y_true, y_pred) if len(y_true) else float("nan")
    f1 = f1_score(y_true, y_pred, zero_division=0) if len(y_true) else float("nan")
    return {
        "loss": total_loss / max(1, total),
        "loss_choice": total_choice / max(1, total),
        "loss_meas": total_meas / max(1, total),
        "acc": acc,
        "f1": f1,
        "log_likelihood": total_ll,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main():
    parser = argparse.ArgumentParser(description="ICLV multimodal (tab + img_emb proyectado + EEG_emb como indicador).")
    parser.add_argument("--data", type=Path, default=Path("./data/processed/multimodal_join_with_eeg_emb.csv"))
    parser.add_argument("--label-col", type=str, default="bought")
    parser.add_argument("--obs-lt-cols", type=str, default="./utils/columns/iclv/obs_lt.txt")
    parser.add_argument("--obs-u-cols", type=str, default="./utils/columns/iclv/obs_u.txt")
    parser.add_argument("--img-emb-col", type=str, default="embedding_path")
    parser.add_argument("--eeg-emb-col", type=str, default="eeg_emb_path")
    parser.add_argument("--num-choices", type=int, default=2)
    parser.add_argument("--n-latent", type=int, default=3)
    parser.add_argument("--img-proj-dim", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0, help="Peso de la pérdida de medición (EEG recon).")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--balance", action="store_true", help="Balancear clases con pos_weight en pérdida de elección.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=Path("./results/multimodal_icl_v"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.lower()
    label_col = args.label_col.lower()
    img_emb_col = args.img_emb_col.lower()
    eeg_emb_col = args.eeg_emb_col.lower()
    df = df.dropna(subset=[label_col, img_emb_col, eeg_emb_col])

    drop_cols = {label_col}
    obs_lt_cols = resolve_cols(df, args.obs_lt_cols, fallback_numeric=False, drop_cols=drop_cols)
    obs_u_cols = resolve_cols(df, args.obs_u_cols, fallback_numeric=True, drop_cols=drop_cols)

    train_df, val_df = split_train_val(df, label_col=label_col, val_split=args.val_split, seed=args.seed)

    # Estandarizar obs_lt y obs_u numéricas en train y aplicar a val
    def standardize_cols(df_fit, df_apply, cols):
        means = df_fit[cols].mean()
        stds = df_fit[cols].std().replace(0, 1)
        return (df_apply[cols] - means) / stds

    train_df = train_df.copy()
    val_df = val_df.copy()
    if obs_lt_cols:
        train_df[obs_lt_cols] = standardize_cols(train_df, train_df, obs_lt_cols)
        val_df[obs_lt_cols] = standardize_cols(train_df, val_df, obs_lt_cols)
    if obs_u_cols:
        train_df[obs_u_cols] = standardize_cols(train_df, train_df, obs_u_cols)
        val_df[obs_u_cols] = standardize_cols(train_df, val_df, obs_u_cols)

    train_ds = MultimodalICLVDataset(train_df, obs_lt_cols, obs_u_cols, label_col, img_emb_col, eeg_emb_col, num_choices=args.num_choices)
    val_ds = MultimodalICLVDataset(val_df, obs_lt_cols, obs_u_cols, label_col, img_emb_col, eeg_emb_col, num_choices=args.num_choices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Infer dims
    sample = train_ds[0]
    dim_obs_lt = sample[0].shape[-1]
    dim_obs_u = sample[1].shape[-1]
    dim_eeg_emb = sample[2].shape[-1]
    dim_img_emb = sample[3].shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalICLVDeterministic(
        dim_obs_lt=dim_obs_lt,
        dim_obs_u=dim_obs_u,
        dim_img_emb=dim_img_emb,
        dim_eeg_emb=dim_eeg_emb,
        n_latent=args.n_latent,
        n_choices=args.num_choices,
        alpha=args.alpha,
        img_proj_dim=args.img_proj_dim,
    ).to(device)
    # pos_weight si balance
    pos_weight = None
    if args.balance:
        pos = (train_df[label_col] == 1).sum()
        neg = (train_df[label_col] == 0).sum()
        if pos > 0:
            pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = None
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, train=True, optimizer=optim, alpha=args.alpha, pos_weight=pos_weight)
        val = run_epoch(model, val_loader, device, train=False, optimizer=None, alpha=args.alpha, pos_weight=pos_weight)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"tr_loss={tr['loss']:.4f} tr_acc={tr['acc']:.3f} tr_f1={tr['f1']:.3f} "
            f"val_loss={val['loss']:.4f} val_acc={val['acc']:.3f} val_f1={val['f1']:.3f}"
        )
        if best_val is None or val["loss"] < best_val["loss"]:
            best_val = val
            torch.save(model.state_dict(), args.results_dir / "best_model.pt")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.results_dir / "model_last.pt")
    with open(args.results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train": {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in tr.items() if k in ["loss", "acc", "f1", "loss_choice", "loss_meas"]},
                "val": {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in val.items() if k in ["loss", "acc", "f1", "loss_choice", "loss_meas"]},
                "best_val_loss": best_val["loss"] if best_val else None,
                "obs_lt_cols": obs_lt_cols,
                "obs_u_cols": obs_u_cols,
                "img_emb_col": img_emb_col,
                "eeg_emb_col": eeg_emb_col,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Guardado en {args.results_dir}")


if __name__ == "__main__":
    main()
