from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open_clip
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.splits import split_by_participant, format_split_report
from utils.metrics_eval import classification_report_basic, save_metrics
from utils.results_io import ensure_dir
from PIL import Image


def load_frame(path: Path) -> Image.Image:
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        return Image.open(path).convert("RGB")
    try:
        t = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        t = torch.load(path, map_location="cpu")
    if isinstance(t, dict):
        for k in ("frames", "video", "x"):
            if k in t:
                t = t[k]
                break
    if not isinstance(t, torch.Tensor):
        raise ValueError(f"No se pudo cargar tensor de {path}")
    if t.dim() == 4:
        t = t[0]
    elif t.dim() != 3:
        raise ValueError(f"Dimensión no soportada: {t.shape}")
    if t.max() <= 1.0:
        t = (t * 255.0).clamp(0, 255)
    t = t.byte()
    img = Image.fromarray(t.permute(1, 2, 0).cpu().numpy())
    return img


class ClipDataset(Dataset):
    def __init__(self, df: pd.DataFrame, path_col: str, label_col: str, preprocess, device: torch.device):
        self.df = df.reset_index(drop=True)
        self.path_col = path_col
        self.label_col = label_col
        self.preprocess = preprocess
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        p = Path(str(row[self.path_col]))
        img = load_frame(p)
        x = self.preprocess(img)
        y = int(row[self.label_col])
        return x, y


def collate_clip(batch):
    xs, ys = zip(*batch)
    X = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y


class ClipClassifier(nn.Module):
    def __init__(self, model_name: str, pretrained: str, num_classes: int, device: torch.device):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.head = nn.Linear(self.model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            z = self.model.encode_image(x)
        return self.head(z)


def main():
    ap = argparse.ArgumentParser(description="Clasificación video con CLIP congelado + head lineal, split por participante.")
    ap.add_argument("--pkl", type=str, default="data/processed/multimodal_av_join_audio_cached.pkl")
    ap.add_argument("--path-col", type=str, default="frames_route")
    ap.add_argument("--label-col", type=str, default="action_proc")
    ap.add_argument("--participant-col", type=str, default="participant")
    ap.add_argument("--model", type=str, default="RN50")
    ap.add_argument("--pretrained", type=str, default="openai")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_pickle(args.pkl).reset_index(drop=True)
    df_tr, df_val, df_te, info = split_by_participant(
        df, participant_col=args.participant_col, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    print(format_split_report(info))

    if df_tr[args.label_col].dtype == object:
        mapping = {v: i for i, v in enumerate(sorted(df[args.label_col].dropna().unique()))}
        for d in (df_tr, df_val, df_te):
            d[args.label_col] = d[args.label_col].map(mapping)
    num_classes = int(pd.Series(df_tr[args.label_col]).nunique())

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = ClipClassifier(model_name=args.model, pretrained=args.pretrained, num_classes=num_classes, device=device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def run_epoch(df_split, train_flag: bool):
        if train_flag:
            model.train()
        else:
            model.eval()
        dataset = ClipDataset(df_split, args.path_col, args.label_col, preprocess=model.preprocess, device=device)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=train_flag, collate_fn=collate_clip)
        ys, preds, logps = [], [], []
        total_loss = 0.0
        for batch in loader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            if train_flag:
                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=args.grad_clip)
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            with torch.no_grad():
                logp = torch.log_softmax(logits, dim=1)
                logps.append(logp.cpu())
                ys.append(y.cpu())
                preds.append(logp.argmax(dim=1).cpu())
        if not ys:
            return {}
        y_true = torch.cat(ys).numpy()
        y_pred = torch.cat(preds).numpy()
        logp_np = torch.cat(logps).numpy()
        metrics = classification_report_basic(y_true, y_pred, log_probs=logp_np)
        metrics["loss"] = total_loss / len(y_true)
        return metrics

    history = []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(df_tr, True)
        val = run_epoch(df_val, False)
        history.append({"epoch": epoch, "train": tr, "val": val})
        print(f"Epoch {epoch:03d} | train acc={tr.get('acc',0):.3f} val acc={val.get('acc',0):.3f}")

    test_metrics = run_epoch(df_te, False) if len(df_te) else {}
    results_dir = Path("results")
    ensure_dir(results_dir)
    base_config = {
        "pkl": args.pkl,
        "path_col": args.path_col,
        "label_col": args.label_col,
        "participant_col": args.participant_col,
        "model": args.model,
        "pretrained": args.pretrained,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "seed": args.seed,
        "argv": sys.argv,
    }
    all_metrics = {f"val_{k}": v for k, v in (history[-1]["val"] or {}).items()}
    all_metrics.update({f"test_{k}": v for k, v in (test_metrics or {}).items()})
    save_metrics(all_metrics, results_dir, model_name="CLIPFreeze", config=base_config)
    hist_path = results_dir / "CLIPFreeze" / "history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    info_path = results_dir / "CLIPFreeze" / "split_info.txt"
    info_path.write_text(format_split_report(info), encoding="utf-8")


if __name__ == "__main__":
    main()
