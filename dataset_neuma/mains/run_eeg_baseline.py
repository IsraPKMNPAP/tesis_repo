"""
Baseline EEG:
- Por defecto usa segmentos crudos (EEGIndexDataset + CNN).
- Si se pasa --embeddings-csv, usa embeddings EEG (eeg_emb_path) con MLP.
Split por sujeto y métricas unificadas.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataloaders.eeg_index import EEGIndexDataset
from src.models.eeg_cnn import EEGCNN
from src.models.tabular_mlp import TabularMLP
from utils.metrics import classification_metrics, save_metrics
from utils.run_utils import save_run_metadata, next_run_dir
from utils.splits import split_by_subject_train_val_test, save_split_info


def collate_to_tensor(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.float32)
    return x, y


class EEGEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, df, emb_col: str = "eeg_emb_path", label_col: str = "bought"):
        self.df = df.reset_index(drop=True)
        self.emb_col = emb_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        emb = np.load(row[self.emb_col]).astype(np.float32)
        if emb.ndim > 1:
            emb = emb.flatten()
        x = torch.tensor(emb, dtype=torch.float32)
        y = torch.tensor(row[self.label_col], dtype=torch.float32)
        return x, y


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()
            ys.append(np.asarray(y))
            ps.append(prob)
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    return y_true, y_prob


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline EEG CNN o MLP sobre embeddings.")
    parser.add_argument("--index-csv", type=Path, default=Path("./data/processed/eeg_segments_index.csv"))
    parser.add_argument("--embeddings-csv", type=Path, default=None)
    parser.add_argument("--emb-col", type=str, default="eeg_emb_path")
    parser.add_argument("--results-dir", type=Path, default=Path("./results/eeg_baseline"))
    parser.add_argument("--segment-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--emb-hidden", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.embeddings_csv:
        df = pd.read_csv(args.embeddings_csv)
        df.columns = df.columns.str.lower()
        emb_col = args.emb_col.lower()
        if "subject" not in df.columns:
            raise SystemExit("El embeddings CSV debe contener columna 'subject' para split por sujeto.")
        df = df.dropna(subset=[emb_col, "bought"])

        train_df, val_df, test_df, split_info = split_by_subject_train_val_test(
            df, subject_col="subject", val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
        )
        print(
            f"[split] subjects={split_info['n_subjects']} train={split_info['n_train_subjects']} "
            f"val={split_info['n_val_subjects']} test={split_info['n_test_subjects']} | "
            f"rows train={split_info['train_rows']} val={split_info['val_rows']} test={split_info['test_rows']}"
        )

        train_ds = EEGEmbeddingDataset(train_df, emb_col=emb_col, label_col="bought")
        val_ds = EEGEmbeddingDataset(val_df, emb_col=emb_col, label_col="bought")
        test_ds = EEGEmbeddingDataset(test_df, emb_col=emb_col, label_col="bought")

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        sample_x, _ = train_ds[0]
        model = TabularMLP(input_dim=sample_x.shape[0], hidden_dims=args.emb_hidden).to(device)
    else:
        full_ds = EEGIndexDataset(index_csv=args.index_csv, segment_len=args.segment_len, cache=True)
        if "subject" not in full_ds.df.columns:
            raise SystemExit("El index CSV debe contener columna 'subject' para split por sujeto.")

        train_df, val_df, test_df, split_info = split_by_subject_train_val_test(
            full_ds.df, subject_col="subject", val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
        )
        print(
            f"[split] subjects={split_info['n_subjects']} train={split_info['n_train_subjects']} "
            f"val={split_info['n_val_subjects']} test={split_info['n_test_subjects']} | "
            f"rows train={split_info['train_rows']} val={split_info['val_rows']} test={split_info['test_rows']}"
        )

        train_idx = train_df.index.to_numpy()
        val_idx = val_df.index.to_numpy()
        test_idx = test_df.index.to_numpy()

        train_ds = Subset(full_ds, train_idx)
        val_ds = Subset(full_ds, val_idx)
        test_ds = Subset(full_ds, test_idx)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_to_tensor)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_to_tensor)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_to_tensor)

        sample_x, _ = full_ds[0]
        in_channels = sample_x.shape[0]
        model = EEGCNN(in_channels=in_channels, hidden=args.hidden).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        y_true_val, y_prob_val = evaluate(model, val_loader, device)
        metrics_val = classification_metrics(y_true_val, y_prob_val)
        print(
            f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} "
            f"val_acc={metrics_val['acc']:.4f} val_f1={metrics_val['f1_macro']:.4f} val_auc={metrics_val['auc']:.4f}"
        )

    y_true_tr, y_prob_tr = evaluate(model, train_loader, device)
    y_true_val, y_prob_val = evaluate(model, val_loader, device)
    y_true_te, y_prob_te = evaluate(model, test_loader, device)

    run_dir = next_run_dir(args.results_dir)
    torch.save(model.state_dict(), run_dir / "model.pt")
    save_split_info(split_info, run_dir)
    save_run_metadata(args, run_dir)

    metrics = {
        "train": classification_metrics(y_true_tr, y_prob_tr),
        "val": classification_metrics(y_true_val, y_prob_val),
        "test": classification_metrics(y_true_te, y_prob_te),
        "loss_final": float(train_loss),
    }
    save_metrics(metrics, run_dir)
    print("Final metrics:", metrics)


if __name__ == "__main__":
    main()
