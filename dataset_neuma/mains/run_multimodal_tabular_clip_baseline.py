"""
Baseline multimodal (tabular + embeddings CLIP) con split por sujeto.
Guarda modelo, preprocessors, split_info y metrics por run.
"""

from __future__ import annotations

import argparse
import json
import re
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

from src.dataloaders.multimodal_tabular_image import load_tabular_image, save_preprocessors
from src.models.tabular_image_fusion import TabImageFusion
from utils.metrics import classification_metrics, save_metrics
from utils.run_utils import save_run_metadata, next_run_dir
from utils.splits import split_by_subject_train_val_test, save_split_info


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
    return y_true, y_prob


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline tabular + imagen (embeddings CLIP).")
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
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--premerged", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    prod_df = pd.read_csv(args.products)
    prod_df.columns = prod_df.columns.str.lower()
    emb_index = pd.read_csv(args.embeddings_dir / "embeddings_index.csv")
    emb_index.columns = emb_index.columns.str.lower()
    tab_df = pd.read_csv(args.tabular)
    tab_df.columns = tab_df.columns.str.lower()
    if "id_sub" in tab_df.columns:
        tab_df = tab_df.rename(columns={"id_sub": "subject"})

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cat_cols = [c.lower() for c in cfg["cat_cols"]]
    num_cols = [c.lower() for c in cfg["num_cols"]]
    label_col = cfg.get("label_col", "bought").lower()

    def page_num(val: str):
        m = re.match(r"page(\d+)", str(val).lower())
        return int(m.group(1)) if m else None

    def prod_num(val: str):
        m = re.match(r"product(\d+)", str(val).lower())
        return int(m.group(1)) if m else None

    def subj_num(val: str) -> str:
        m = re.match(r"s0*(\d+)", str(val).lower())
        return m.group(1) if m else str(val)

    merged = prod_df.merge(emb_index[["page", "product_id", "embedding_path"]], on=["page", "product_id"], how="left")

    if label_col not in merged.columns:
        candidates = [c for c in merged.columns if c.startswith(label_col)]
        if candidates:
            merged = merged.rename(columns={candidates[0]: label_col})
        else:
            raise SystemExit(f"No se encontro columna de etiqueta '{label_col}' en products.")

    merged["subject_norm"] = merged["subject"].apply(subj_num)

    if not args.premerged:
        if "subject" not in tab_df.columns:
            raise SystemExit("La tabla tabular no contiene columna 'subject'.")
        tab_df["subject_norm"] = tab_df["subject"].apply(lambda s: str(int(s)) if str(s).isdigit() else str(s))
        merged["page_num"] = merged["page"].apply(page_num)
        merged["prod_num"] = merged["product_id"].apply(prod_num)
        merged["id_prod_key"] = merged.apply(
            lambda r: 24 * (r["page_num"] - 1) + r["prod_num"] if pd.notna(r["page_num"]) and pd.notna(r["prod_num"]) else np.nan,
            axis=1,
        )
        if "id_prod" in tab_df.columns:
            merged = merged.merge(
                tab_df,
                left_on=["subject_norm", "id_prod_key"],
                right_on=["subject_norm", "id_prod"],
                how="left",
                suffixes=("", "_tab"),
            )
        else:
            raise SystemExit("La tabla tabular no contiene columna 'id_prod'.")
        merged = merged.dropna(subset=["embedding_path", label_col])
        missing_cols = [c for c in (cat_cols + num_cols) if c not in merged.columns]
        if missing_cols:
            raise SystemExit(f"Faltan columnas tabulares en el merge: {missing_cols}")
        merged = merged.dropna(subset=cat_cols + num_cols + [label_col])
    else:
        merged = merged.dropna(subset=["embedding_path", label_col, "subject"])

    merged[label_col] = merged[label_col].astype(int)

    full_df = merged.reset_index(drop=True)
    train_df, val_df, test_df, split_info = split_by_subject_train_val_test(
        full_df, subject_col="subject", val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    print(
        f"[split] subjects={split_info['n_subjects']} train={split_info['n_train_subjects']} "
        f"val={split_info['n_val_subjects']} test={split_info['n_test_subjects']} | "
        f"rows train={split_info['train_rows']} val={split_info['val_rows']} test={split_info['test_rows']}"
    )

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
    test_loader, _, _, _, _ = load_tabular_image(
        test_df,
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
    save_preprocessors(ohe, scaler, run_dir)
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
