"""
Baseline multimodal (tab + CLIP + EEG) con cabeza de logits interpretables.
Split por sujeto, métricas unificadas y versionado incremental.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.dataloaders.multimodal_all import MultimodalDataset, prepare_feature_lists
from src.models.multimodal_fusion_logits import FusionClassifierLogits, FusionVAELogits
from utils.metrics import classification_metrics, save_metrics
from utils.run_utils import save_run_metadata, next_run_dir
from utils.splits import split_by_subject_train_val_test, save_split_info


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
            bce = criterion(logits, y if logits.ndim == 1 else y.long())
            kl = model.kl_div(mu, logvar).mean()
            loss = bce + beta_kl * kl
        else:
            logits = model(tab, clip, eeg)
            bce = criterion(logits, y if logits.ndim == 1 else y.long())
            loss = bce
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
    logit_dim: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
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
            if logit_dim == 1:
                prob = torch.sigmoid(logits).cpu().numpy()
            else:
                prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            ys.append(np.asarray(y))
            ps.append(prob)
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    return y_true, y_prob


def parse_thresholds(thr_list: str):
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
    parser = argparse.ArgumentParser(description="Multimodal con cabeza de logits interpretables.")
    parser.add_argument("--data", type=Path, default=Path("./data/processed/multimodal_join.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("./results/multimodal_logits_baseline"))
    parser.add_argument("--mode", type=str, default="deterministic", choices=["deterministic", "vae"])
    parser.add_argument("--use-tabular", action="store_true")
    parser.add_argument("--use-clip", action="store_true")
    parser.add_argument("--use-eeg", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eeg-len", type=int, default=2048)
    parser.add_argument("--img-proj", type=int, default=0)
    parser.add_argument("--beta-kl", type=float, default=1e-3)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--thresholds", type=str, default="0.5,0.4,0.6")
    parser.add_argument("--logit-dim", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    full_df = pd.read_csv(args.data)
    full_df.columns = full_df.columns.str.lower()
    full_df = full_df.dropna(subset=["embedding_path", "eeg_concat_path", "bought"])
    full_df["bought"] = full_df["bought"].astype(int)
    if "subject" not in full_df.columns:
        raise SystemExit("Se requiere columna 'subject' para split por sujeto.")

    train_df, val_df, test_df, split_info = split_by_subject_train_val_test(
        full_df, subject_col="subject", val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    print(
        f"[split] subjects={split_info['n_subjects']} train={split_info['n_train_subjects']} "
        f"val={split_info['n_val_subjects']} test={split_info['n_test_subjects']} | "
        f"rows train={split_info['train_rows']} val={split_info['val_rows']} test={split_info['test_rows']}"
    )

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
    test_loader, _, _, _, _, _ = make_loader(test_df, ohe=ohe, scaler=scaler, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "vae":
        model = FusionVAELogits(
            tab_dim=tab_dim if args.use_tabular else 0,
            clip_dim=clip_dim if args.use_clip else 0,
            eeg_ch=eeg_ch if args.use_eeg else 0,
            img_proj=args.img_proj,
            logit_dim=args.logit_dim,
        ).to(device)
    else:
        model = FusionClassifierLogits(
            tab_dim=tab_dim if args.use_tabular else 0,
            clip_dim=clip_dim if args.use_clip else 0,
            eeg_ch=eeg_ch if args.use_eeg else 0,
            img_proj=args.img_proj,
            logit_dim=args.logit_dim,
        ).to(device)

    pos_weight = None
    if args.balance and args.logit_dim == 1:
        pos = (train_df["bought"] == 1).sum()
        neg = (train_df["bought"] == 0).sum()
        if pos > 0:
            pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    if args.logit_dim == 1:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.CrossEntropyLoss()

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
        y_true_val, y_prob_val = evaluate(
            model,
            val_loader,
            device,
            mode=args.mode,
            use_tab=args.use_tabular,
            use_clip=args.use_clip,
            use_eeg=args.use_eeg,
            logit_dim=args.logit_dim,
        )
        metrics_val = classification_metrics(y_true_val, y_prob_val)
        print(
            f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} "
            f"val_acc={metrics_val['acc']:.4f} val_f1={metrics_val['f1_macro']:.4f} val_auc={metrics_val['auc']:.4f}"
        )

    y_true_tr, y_prob_tr = evaluate(model, train_loader, device, mode=args.mode, use_tab=args.use_tabular, use_clip=args.use_clip, use_eeg=args.use_eeg, logit_dim=args.logit_dim)
    y_true_val, y_prob_val = evaluate(model, val_loader, device, mode=args.mode, use_tab=args.use_tabular, use_clip=args.use_clip, use_eeg=args.use_eeg, logit_dim=args.logit_dim)
    y_true_te, y_prob_te = evaluate(model, test_loader, device, mode=args.mode, use_tab=args.use_tabular, use_clip=args.use_clip, use_eeg=args.use_eeg, logit_dim=args.logit_dim)

    run_dir = next_run_dir(args.results_dir)
    torch.save(model.state_dict(), run_dir / f"model_{args.mode}.pt")
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
