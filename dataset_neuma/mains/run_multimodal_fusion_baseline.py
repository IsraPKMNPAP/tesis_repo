"""
Entrena un baseline multimodal con fusión temprana de tres modalidades:
  - Tabular (profiles + metadata de products_all_with_images)
  - Embeddings CLIP precomputados (congelados; se usan tal cual)
  - EEG concatenado por (subject, page, product)

Usa el CSV construido por build_multimodal_join.py (multimodal_join.csv).
Permite dos modos:
  --mode deterministic (MLP)
  --mode vae (variational; usa KL + BCE)

Guarda modelo y métricas en results/multimodal_fusion_baseline.

Uso (desde dataset_neuma):
  python -m mains.run_multimodal_fusion_baseline \
    --data ./data/processed/multimodal_join.csv \
    --results-dir ./results/multimodal_fusion_baseline \
    --mode deterministic \
    --batch-size 32 --epochs 10 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Permite ejecución desde carpeta dataset_neuma
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataloaders.multimodal_all import MultimodalDataset, prepare_feature_lists
from src.models.multimodal_fusion import FusionClassifier, FusionVAE


def collate_fn(batch):
    tabs, clips, eegs, ys = zip(*batch)
    return (
        torch.stack(tabs, dim=0),
        torch.stack(clips, dim=0),
        torch.stack(eegs, dim=0),
        torch.tensor(ys, dtype=torch.float32),
    )


def train_one_epoch(model, loader, optimizer, criterion, device, mode="deterministic", beta_kl: float = 1e-3):
    model.train()
    total_loss = 0.0
    for tab, clip, eeg, y in loader:
        tab, clip, eeg, y = tab.to(device), clip.to(device), eeg.to(device), y.to(device)
        optimizer.zero_grad()
        if mode == "vae":
            logits, mu, logvar = model(tab, clip, eeg)
            bce = criterion(logits, y)
            kl = model.kl_div(mu, logvar).mean()
            loss = bce + beta_kl * kl
        else:
            logits = model(tab, clip, eeg)
            loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, mode="deterministic"):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for tab, clip, eeg, y in loader:
            tab, clip, eeg = tab.to(device), clip.to(device), eeg.to(device)
            if mode == "vae":
                logits, _, _ = model(tab, clip, eeg)
            else:
                logits = model(tab, clip, eeg)
            prob = torch.sigmoid(logits).cpu().numpy()
            ys.append(np.asarray(y))
            ps.append(prob)
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return acc, f1, auc


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline multimodal (tab + clip + eeg) con fusión temprana.")
    parser.add_argument("--data", type=Path, default=Path("./data/processed/multimodal_join.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("./results/multimodal_fusion_baseline"))
    parser.add_argument("--mode", type=str, default="deterministic", choices=["deterministic", "vae"])
    parser.add_argument("--use-tabular", action="store_true", help="Usar tabular (si no, se anula tab).")
    parser.add_argument("--use-clip", action="store_true", help="Usar embeddings CLIP (si no, se anula clip).")
    parser.add_argument("--use-eeg", action="store_true", help="Usar EEG (si no, se anula EEG).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-by-subject", action="store_true", help="Si se activa, split se hace por sujeto (subject_norm) para evitar fuga directa.")
    parser.add_argument("--eeg-len", type=int, default=2048)
    parser.add_argument("--img-proj", type=int, default=0, help="Si >0, proyecta el embedding CLIP a esa dimensión.")
    parser.add_argument("--beta-kl", type=float, default=1e-3, help="Peso del término KL en modo VAE.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    full_df = pd.read_csv(args.data)
    full_df.columns = full_df.columns.str.lower()
    full_df = full_df.dropna(subset=["embedding_path", "eeg_concat_path", "bought"])
    full_df["bought"] = full_df["bought"].astype(int)

    if args.split_by_subject and "subject" in full_df.columns:
        subjects = full_df["subject"].unique()
        train_subj, val_subj = train_test_split(subjects, test_size=args.val_size, random_state=args.seed)
        train_df = full_df[full_df["subject"].isin(train_subj)].reset_index(drop=True)
        val_df = full_df[full_df["subject"].isin(val_subj)].reset_index(drop=True)
    else:
    if args.split_by_subject and "subject" in full_df.columns:
        subjects = full_df["subject"].unique()
        train_subj, val_subj = train_test_split(subjects, test_size=args.val_size, random_state=args.seed)
        train_df = full_df[full_df["subject"].isin(train_subj)].reset_index(drop=True)
        val_df = full_df[full_df["subject"].isin(val_subj)].reset_index(drop=True)
    else:
        y_all = full_df["bought"].to_numpy()
        idxs = np.arange(len(full_df))
        train_idx, val_idx = train_test_split(idxs, test_size=args.val_size, random_state=args.seed, stratify=y_all)
        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_df.iloc[val_idx].reset_index(drop=True)

    def make_loader(df: pd.DataFrame, ohe=None, scaler=None, shuffle=True):
        # quitar cualquier columna que contenga 'bought' salvo la etiqueta principal
        drop_cols = [c for c in df.columns if ("bought" in c) and (c != "bought")]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        cat_cols, num_cols = prepare_feature_lists(df, "bought")
        ds = MultimodalDataset(
            df=df,
            cat_cols=cat_cols,
            num_cols=num_cols,
            label_col="bought",
            eeg_len=args.eeg_len,
            cache_clip=True,
            cache_eeg=True,
            ohe=ohe,
            scaler=scaler,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_fn)
        return loader, ds.ohe, ds.scaler, ds.tab.shape[1], len(np.load(ds.clip_paths[0])), np.load(ds.eeg_paths[0]).shape[0]

    train_loader, ohe, scaler, tab_dim, clip_dim, eeg_ch = make_loader(train_df, ohe=None, scaler=None, shuffle=True)
    val_loader, _, _, _, _, _ = make_loader(val_df, ohe=ohe, scaler=scaler, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "vae":
        model = FusionVAE(tab_dim=tab_dim if args.use_tabular else 0, clip_dim=clip_dim if args.use_clip else 0, eeg_ch=eeg_ch if args.use_eeg else 0, img_proj=args.img_proj).to(device)
    else:
        model = FusionClassifier(tab_dim=tab_dim if args.use_tabular else 0, clip_dim=clip_dim if args.use_clip else 0, eeg_ch=eeg_ch if args.use_eeg else 0, img_proj=args.img_proj).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, mode=args.mode, beta_kl=args.beta_kl)
        acc, f1, auc = evaluate(model, val_loader, device, mode=args.mode)
        print(f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

    acc, f1, auc = evaluate(model, val_loader, device, mode=args.mode)
    metrics = {"acc": acc, "f1": f1, "auc": auc}

    args.results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.results_dir / f"model_{args.mode}.pt")
    with open(args.results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Final metrics:", metrics)


if __name__ == "__main__":
    main()
