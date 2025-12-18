"""
Baseline EEG: CNN simple sobre segmentos crudos para predecir bought.

Usa el índice agregado:
  data/processed/eeg_segments_index.csv
que contiene npy_path, start, end, bought (ya mergeado por producto).

Entrenamiento:
  - Recorta/pad cada segmento a longitud fija (segment_len).
  - Split train/val (stratificado).
  - Métricas: acc, f1, auc.
Guarda:
  - modelo (.pt), métricas (.json) en results/eeg_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# Permite ejecución desde carpeta dataset_neuma
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataloaders.eeg_index import EEGIndexDataset
from src.models.eeg_cnn import EEGCNN


def collate_to_tensor(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)  # [B, C, T]
    y = torch.tensor(ys, dtype=torch.float32)
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
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return acc, f1, auc


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline EEG CNN (segmentos crudos).")
    parser.add_argument("--index-csv", type=Path, default=Path("./data/processed/eeg_segments_index.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("./results/eeg_baseline"))
    parser.add_argument("--segment-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    full_ds = EEGIndexDataset(index_csv=args.index_csv, segment_len=args.segment_len, cache=True)
    n = len(full_ds)
    idxs = np.arange(n)
    y_all = full_ds.df["bought"].to_numpy()
    train_idx, val_idx = train_test_split(
        idxs, test_size=args.val_size, random_state=args.seed, stratify=y_all
    )
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_to_tensor)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_to_tensor)

    # Infer channel count from one sample
    sample_x, _ = full_ds[0]
    in_channels = sample_x.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGCNN(in_channels=in_channels, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc, f1, auc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

    # Final eval en val
    acc, f1, auc = evaluate(model, val_loader, device)
    metrics = {"acc": acc, "f1": f1, "auc": auc}

    args.results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.results_dir / "model.pt")
    (args.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Final metrics:", metrics)


if __name__ == "__main__":
    main()

