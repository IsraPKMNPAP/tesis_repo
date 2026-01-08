"""
Baseline tabular: MLP para predecir bought con split por sujeto.
Guarda modelo, preprocessors, split_info y metrics por run.
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
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataloaders.tabular import TabularDataset, save_preprocessors
from src.models.tabular_mlp import TabularMLP
from utils.metrics import classification_metrics, save_metrics
from utils.run_utils import save_run_metadata, next_run_dir
from utils.splits import split_by_subject_train_val_test, save_split_info


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
            ys.append(y.numpy())
            ps.append(prob)
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    return y_true, y_prob


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline tabular MLP para NEUMA.")
    parser.add_argument("--data", type=Path, default=Path("/mnt/otra_particion/home/israel_gpu_data/dataset_neuma/processed/data_base_neuma.csv"))
    parser.add_argument("--config", type=Path, default=Path("dataset_neuma/configs/tabular_cols.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("dataset_neuma/results/tabular_baseline"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cat_cols = [c.lower() for c in cfg["cat_cols"]]
    num_cols = [c.lower() for c in cfg["num_cols"]]
    label_col = cfg.get("label_col", "bought").lower()

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.lower()
    if "subject" not in df.columns and "id_sub" in df.columns:
        df["subject"] = df["id_sub"].astype(str)

    train_df, val_df, test_df, split_info = split_by_subject_train_val_test(
        df, subject_col="subject", val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    print(
        f"[split] subjects={split_info['n_subjects']} train={split_info['n_train_subjects']} "
        f"val={split_info['n_val_subjects']} test={split_info['n_test_subjects']} | "
        f"rows train={split_info['train_rows']} val={split_info['val_rows']} test={split_info['test_rows']}"
    )

    train_ds = TabularDataset(train_df, cat_cols, num_cols, label_col)
    val_ds = TabularDataset(val_df, cat_cols, num_cols, label_col, ohe=train_ds.ohe, scaler=train_ds.scaler)
    test_ds = TabularDataset(test_df, cat_cols, num_cols, label_col, ohe=train_ds.ohe, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularMLP(input_dim=train_ds.x.shape[1], hidden_dims=args.hidden).to(device)
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
    save_preprocessors(train_ds.ohe, train_ds.scaler, run_dir)
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
