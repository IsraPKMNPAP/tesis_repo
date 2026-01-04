"""
Baseline multimodal con fusión temprana (tabular + CLIP + EEG).

Usa el CSV generado por build_multimodal_join.py (multimodal_join.csv) con
embeddings de producto, perfiles tabulares y EEG concatenado por (subject, page, product).

Modos:
  - deterministic: MLP simple
  - vae: encoder probabilístico con KL

Se permite activar/desactivar cada modalidad y balancear clases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Ejecutable desde dataset_neuma
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


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    mode: str = "deterministic",
    beta_kl: float = 1e-3,
    use_tab: bool = True,
    use_clip: bool = True,
    use_eeg: bool = True,
):
    model.train()
    total_loss = 0.0
    for tab, clip, eeg, y in loader:
        tab, clip, eeg, y = tab.to(device), clip.to(device), eeg.to(device), y.to(device)
        if not use_tab:
            tab = torch.zeros(tab.shape[0], 0, device=device, dtype=tab.dtype)
        if not use_clip:
            clip = torch.zeros(clip.shape[0], 0, device=device, dtype=clip.dtype)
        if not use_eeg:
            eeg = torch.zeros(eeg.shape[0], 0, device=device, dtype=eeg.dtype)

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


def evaluate(
    model,
    loader,
    device,
    mode: str = "deterministic",
    use_tab: bool = True,
    use_clip: bool = True,
    use_eeg: bool = True,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for tab, clip, eeg, y in loader:
            tab, clip, eeg = tab.to(device), clip.to(device), eeg.to(device)
            if not use_tab:
                tab = torch.zeros(tab.shape[0], 0, device=device, dtype=tab.dtype)
            if not use_clip:
                clip = torch.zeros(clip.shape[0], 0, device=device, dtype=clip.dtype)
            if not use_eeg:
                eeg = torch.zeros(eeg.shape[0], 0, device=device, dtype=eeg.dtype)
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
    return acc, f1, auc, y_true, y_prob


def parse_thresholds(thr_list: str) -> Iterable[float]:
    vals = []
    for part in thr_list.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(float(part))
        except ValueError:
            continue
    return vals if vals else [0.5]


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline multimodal (tab+clip+eeg) con fusión temprana.")
    parser.add_argument("--data", type=Path, default=Path("./data/processed/multimodal_join.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("./results/multimodal_fusion_baseline"))
    parser.add_argument("--mode", type=str, default="deterministic", choices=["deterministic", "vae"])
    parser.add_argument("--use-tabular", action="store_true", help="Usar tabular (si no, se anula tab).")
    parser.add_argument("--use-clip", action="store_true", help="Usar embeddings CLIP (si no, se anula clip).")
    parser.add_argument("--use-eeg", action="store_true", help="Usar EEG (si no, se anula EEG).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-by-subject", action="store_true", help="Split por sujeto (evita fuga directa).")
    parser.add_argument("--eeg-len", type=int, default=2048)
    parser.add_argument("--img-proj", type=int, default=0, help="Si >0, proyecta embedding CLIP a esa dimensión.")
    parser.add_argument("--beta-kl", type=float, default=1e-3, help="Peso del término KL en modo VAE.")
    parser.add_argument("--balance", action="store_true", help="BCE con pos_weight según train.")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.5,0.4,0.6",
        help="Lista separada por comas para barrer f1/acc en validación.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    full_df = pd.read_csv(args.data)
    full_df.columns = full_df.columns.str.lower()
    # imputar price/len_med si existen
    for col in ["price", "len_med"]:
        if col in full_df.columns:
            med = full_df[col].median()
            full_df[col] = full_df[col].fillna(med)
    full_df = full_df.dropna(subset=["embedding_path", "eeg_concat_path", "bought"])
    full_df["bought"] = full_df["bought"].astype(int)

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

    thresholds = list(parse_thresholds(args.thresholds))

    def make_loader(df: pd.DataFrame, ohe=None, scaler=None, shuffle: bool = True):
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
        model = FusionVAE(
            tab_dim=tab_dim if args.use_tabular else 0,
            clip_dim=clip_dim if args.use_clip else 0,
            eeg_ch=eeg_ch if args.use_eeg else 0,
            img_proj=args.img_proj,
        ).to(device)
    else:
        model = FusionClassifier(
            tab_dim=tab_dim if args.use_tabular else 0,
            clip_dim=clip_dim if args.use_clip else 0,
            eeg_ch=eeg_ch if args.use_eeg else 0,
            img_proj=args.img_proj,
        ).to(device)

    pos_weight = None
    if args.balance:
        pos = (train_df["bought"] == 1).sum()
        neg = (train_df["bought"] == 0).sum()
        if pos > 0:
            pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            mode=args.mode,
            beta_kl=args.beta_kl,
            use_tab=args.use_tabular,
            use_clip=args.use_clip,
            use_eeg=args.use_eeg,
        )
        acc, f1, auc, y_true, y_prob = evaluate(
            model,
            val_loader,
            device,
            mode=args.mode,
            use_tab=args.use_tabular,
            use_clip=args.use_clip,
            use_eeg=args.use_eeg,
        )
        # barrido de thresholds para ver mejor f1/acc
        best = None
        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(int)
            f1_t = f1_score(y_true, y_pred, zero_division=0)
            acc_t = accuracy_score(y_true, y_pred)
            if (best is None) or (f1_t > best["f1"]):
                best = {"thr": thr, "f1": f1_t, "acc": acc_t}
        best_thr = best["thr"] if best else 0.5
        best_f1 = best["f1"] if best else f1
        best_acc = best["acc"] if best else acc
        print(
            f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} acc@0.5={acc:.4f} "
            f"f1@0.5={f1:.4f} auc={auc:.4f} best_thr={best_thr:.2f} "
            f"best_acc={best_acc:.4f} best_f1={best_f1:.4f}"
        )

    acc, f1, auc, y_true, y_prob = evaluate(
        model,
        val_loader,
        device,
        mode=args.mode,
        use_tab=args.use_tabular,
        use_clip=args.use_clip,
        use_eeg=args.use_eeg,
    )
    best = None
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        f1_t = f1_score(y_true, y_pred, zero_division=0)
        acc_t = accuracy_score(y_true, y_pred)
        if (best is None) or (f1_t > best["f1"]):
            best = {"thr": thr, "f1": f1_t, "acc": acc_t}
    best_thr = best["thr"] if best else 0.5
    best_f1 = best["f1"] if best else f1
    best_acc = best["acc"] if best else acc

    metrics = {
        "acc@0.5": acc,
        "f1@0.5": f1,
        "auc": auc,
        "best_thr": best_thr,
        "best_acc": best_acc,
        "best_f1": best_f1,
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.results_dir / f"model_{args.mode}.pt")
    with open(args.results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Final metrics:", metrics)


if __name__ == "__main__":
    main()
