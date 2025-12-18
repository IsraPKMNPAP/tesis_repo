"""
Baseline multimodal (tabular + embeddings de imágenes CLIP) con fusión temprana.

Datos:
  - products_all_with_images.csv (subject, page, product_id, bought, image_path, etc.)
  - image embeddings: data/processed/image_embeddings/embeddings_index.csv
  - tabular sujetos: /mnt/otra_particion/home/israel_gpu_data/dataset_neuma/processed/data_latente_neuma.csv
  - columnas tabulares: configs/tabular_cols.json

Pipeline:
  - Merge products_with_images con embeddings_index (por page, product_id) -> embedding_path
  - Merge con tabular sujetos (por subject)
  - Filtrar rows con label y embedding_path
  - Dataloader que entrega (tabular_vec, embedding_vec, label)
  - Modelo: proyección lineal de embedding + MLP tabular (fusión temprana)
  - Split train/val estratificado, métricas acc/f1/auc
  - Guarda modelo, preprocessors y métricas en results/multimodal_tabular_image_baseline
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
from torch.utils.data import DataLoader, Subset

# Permite ejecución desde carpeta dataset_neuma
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataloaders.multimodal_tabular_image import load_tabular_image, save_preprocessors, TabularImageDataset
from src.models.tabular_image_fusion import TabImageFusion


def collate_fn(batch):
    tabs, imgs, ys = zip(*batch)
    tab = torch.stack(tabs, dim=0)
    img = torch.stack(imgs, dim=0)
    y = torch.tensor(ys, dtype=torch.float32)
    return tab, img, y


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for tab, img, y in loader:
        tab, img, y = tab.to(device), img.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(tab, img)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for tab, img, y in loader:
            tab, img = tab.to(device), img.to(device)
            logits = model(tab, img)
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
    parser = argparse.ArgumentParser(description="Baseline multimodal tabular + imagen (embeddings CLIP).")
    parser.add_argument("--products", type=Path, default=Path("./data/processed/products_all_with_images.csv"))
    parser.add_argument("--embeddings-dir", type=Path, default=Path("./data/processed/image_embeddings"))
    parser.add_argument("--tabular", type=Path, default=Path("/mnt/otra_particion/home/israel_gpu_data/dataset_neuma/processed/data_latente_neuma.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/tabular_cols.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/multimodal_tabular_image_baseline"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--img-proj", type=int, default=128)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Cargar tablas
    prod_df = pd.read_csv(args.products)
    prod_df.columns = prod_df.columns.str.lower()
    emb_index = pd.read_csv(args.embeddings_dir / "embeddings_index.csv")
    emb_index.columns = emb_index.columns.str.lower()
    tab_df = pd.read_csv(args.tabular)
    tab_df.columns = tab_df.columns.str.lower()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cat_cols = [c.lower() for c in cfg["cat_cols"]]
    num_cols = [c.lower() for c in cfg["num_cols"]]
    label_col = cfg.get("label_col", "bought").lower()

    # Merge products con embeddings
    merged = prod_df.merge(emb_index[["page", "product_id", "embedding_path"]], on=["page", "product_id"], how="left")
    merged = merged.dropna(subset=["embedding_path", label_col, "subject"])

    # Merge con tabular sujeto
    merged = merged.merge(tab_df, on="subject", how="left")
    merged = merged.dropna(subset=cat_cols + num_cols + [label_col])

    merged[label_col] = merged[label_col].astype(int)

    full_df = merged.reset_index(drop=True)
    y_all = full_df[label_col].to_numpy()
    idxs = np.arange(len(full_df))
    train_idx, val_idx = train_test_split(idxs, test_size=args.val_size, random_state=args.seed, stratify=y_all)

    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    val_df = full_df.iloc[val_idx].reset_index(drop=True)

    train_loader, ohe, scaler, tab_dim, emb_dim = load_tabular_image(
        train_df,
        cat_cols,
        num_cols,
        label_col,
        batch_size=args.batch_size,
        cache_embeddings=True,
        shuffle=True,
    )
    val_loader, _, _, _, _ = load_tabular_image(
        val_df,
        cat_cols,
        num_cols,
        label_col,
        batch_size=args.batch_size,
        cache_embeddings=True,
        shuffle=False,
        ohe=ohe,
        scaler=scaler,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabImageFusion(tab_dim=tab_dim, img_dim=emb_dim, img_proj=args.img_proj, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc, f1, auc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

    acc, f1, auc = evaluate(model, val_loader, device)
    metrics = {"acc": acc, "f1": f1, "auc": auc}

    args.results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.results_dir / "model.pt")
    save_preprocessors(ohe, scaler, args.results_dir)
    (args.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Final metrics:", metrics)


if __name__ == "__main__":
    main()
