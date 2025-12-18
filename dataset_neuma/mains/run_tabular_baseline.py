"""
Baseline tabular: MLP para predecir bought usando columns definidas en config.

Lectura:
  - Datos: /mnt/otra_particion/home/israel_gpu_data/dataset_neuma/processed/data_base_neuma.csv
  - Config de columnas: dataset_neuma/configs/tabular_cols.json

Guarda en results:
  - modelo (.pt)
  - preprocessors (ohe/scaler)
  - métrica (acc, f1, auc) en JSON
  - log de entrenamiento (simple)
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
from torch.utils.data import DataLoader

# Permite ejecución desde carpeta dataset_neuma
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataloaders.tabular import load_tabular, save_preprocessors
from src.models.tabular_mlp import TabularMLP


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
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return acc, f1, auc, y_true, y_prob


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline tabular MLP para NEUMA.")
    parser.add_argument("--data", type=Path, default=Path("/mnt/otra_particion/home/israel_gpu_data/dataset_neuma/processed/data_base_neuma.csv"))
    parser.add_argument("--config", type=Path, default=Path("dataset_neuma/configs/tabular_cols.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("dataset_neuma/results/tabular_baseline"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cat_cols = [c.lower() for c in cfg["cat_cols"]]
    num_cols = [c.lower() for c in cfg["num_cols"]]
    label_col = cfg.get("label_col", "bought").lower()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, ohe, scaler, input_dim = load_tabular(args.data, cat_cols, num_cols, label_col, batch_size=args.batch_size, shuffle=True)

    model = TabularMLP(input_dim=input_dim, hidden_dims=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loader, optimizer, criterion, device)
        acc, f1, auc, _, _ = evaluate(model, loader, device)
        print(f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

    acc, f1, auc, y_true, y_prob = evaluate(model, loader, device)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.results_dir / "model.pt")
    save_preprocessors(ohe, scaler, args.results_dir)
    metrics = {"acc": acc, "f1": f1, "auc": auc}
    (args.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Final metrics:", metrics)


if __name__ == "__main__":
    main()
