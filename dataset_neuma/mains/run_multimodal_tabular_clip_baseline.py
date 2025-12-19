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
import re

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
    parser.add_argument("--premerged", action="store_true", help="Indica que el CSV de products ya contiene columnas tabulares (no hace merge).")
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
    # Renombrar ID_sub -> subject si existe
    if "id_sub" in tab_df.columns:
        tab_df = tab_df.rename(columns={"id_sub": "subject"})
    # Forzar subject a string en todos los dataframes
    if "subject" in prod_df.columns:
        prod_df["subject"] = prod_df["subject"].astype(str)
    if "subject" in tab_df.columns:
        tab_df["subject"] = tab_df["subject"].astype(str)
    if "subject" in emb_index.columns:
        emb_index["subject"] = emb_index["subject"].astype(str)

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cat_cols = [c.lower() for c in cfg["cat_cols"]]
    num_cols = [c.lower() for c in cfg["num_cols"]]
    label_col = cfg.get("label_col", "bought").lower()

    # Funciones para parsear PageX/ProductY y subject Sxx
    def page_num(val: str) -> int:
        m = re.match(r"page(\d+)", str(val).lower())
        return int(m.group(1)) if m else None

    def prod_num(val: str) -> int:
        m = re.match(r"product(\d+)", str(val).lower())
        return int(m.group(1)) if m else None

    def subj_num(val: str) -> str:
        m = re.match(r"s0*(\d+)", str(val).lower())
        return m.group(1) if m else str(val)

    # Merge products con embeddings
    merged = prod_df.merge(emb_index[["page", "product_id", "embedding_path"]], on=["page", "product_id"], how="left")

    # Normalizar etiqueta
    if label_col not in merged.columns:
        candidates = [c for c in merged.columns if c.startswith(label_col)]
        if candidates:
            merged = merged.rename(columns={candidates[0]: label_col})
        else:
            raise SystemExit(f"No se encontró la columna de etiqueta '{label_col}' en products; cols: {merged.columns.tolist()}")

    # Si no está subject en products, intenta derivarlo (pero aquí sí está). Formatear para matching con tabular numérica.
    merged["subject_norm"] = merged["subject"].apply(subj_num)

    if not args.premerged:
        # Preparar llaves para tabular: subject y id_prod
        if "subject" not in tab_df.columns:
            raise SystemExit("La tabla tabular no contiene columna 'subject' (ni 'ID_sub' renombrada).")
        tab_df["subject_norm"] = tab_df["subject"].apply(lambda s: str(int(s)) if str(s).isdigit() else str(s))

        # Calcular id_prod en products: 24*(page-1)+product
        merged["page_num"] = merged["page"].apply(page_num)
        merged["prod_num"] = merged["product_id"].apply(prod_num)
        merged["id_prod_key"] = merged.apply(lambda r: 24 * (r["page_num"] - 1) + r["prod_num"] if pd.notna(r["page_num"]) and pd.notna(r["prod_num"]) else np.nan, axis=1)

        if "id_prod" in tab_df.columns:
            merged = merged.merge(
                tab_df,
                left_on=["subject_norm", "id_prod_key"],
                right_on=["subject_norm", "id_prod"],
                how="left",
                suffixes=("", "_tab"),
            )
        else:
            raise SystemExit("La tabla tabular no contiene columna 'id_prod' para la llave de producto.")

        merged = merged.dropna(subset=["embedding_path", label_col])
        # Validar columnas tabulares
        missing_cols = [c for c in (cat_cols + num_cols) if c not in merged.columns]
        if missing_cols:
            raise SystemExit(f"Faltan columnas tabulares en el merge: {missing_cols}")
        merged = merged.dropna(subset=cat_cols + num_cols + [label_col])
    else:
        merged = merged.dropna(subset=["embedding_path", label_col, "subject"])

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
