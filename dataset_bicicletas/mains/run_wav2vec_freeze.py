from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.splits import split_by_participant, format_split_report
from utils.metrics_eval import classification_report_basic, save_metrics
from utils.results_io import ensure_dir


class Wav2VecDataset(Dataset):
    def __init__(self, df: pd.DataFrame, audio_col: str, label_col: str, sr_cache: int, target_sr: int):
        self.df = df.reset_index(drop=True)
        self.audio_col = audio_col
        self.label_col = label_col
        self.sr_cache = sr_cache
        self.target_sr = target_sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = Path(str(row[self.audio_col]))
        try:
            seg = torch.load(path, map_location="cpu")
        except Exception:
            seg = torch.zeros(1, int(self.target_sr * 5))
        if isinstance(seg, list):
            seg = torch.tensor(seg, dtype=torch.float32)
        elif isinstance(seg, torch.Tensor):
            seg = seg.float()
        if seg.dim() == 1:
            seg = seg.unsqueeze(0)
        elif seg.dim() == 2 and seg.size(0) != 1:
            seg = seg.mean(dim=0, keepdim=True)
        if self.sr_cache != self.target_sr:
            seg = torchaudio.functional.resample(seg, self.sr_cache, self.target_sr)
        y = int(row[self.label_col])
        return seg, y


def collate(batch):
    xs, ys = zip(*batch)
    max_len = max(x.shape[-1] for x in xs)
    padded = []
    for x in xs:
        if x.shape[-1] < max_len:
            x = torch.nn.functional.pad(x, (0, max_len - x.shape[-1]))
        padded.append(x)
    X = torch.stack(padded, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y


class Wav2VecClassifier(nn.Module):
    def __init__(self, bundle_name: str, num_classes: int, dropout: float = 0.1, trainable: bool = False):
        super().__init__()
        bundle = getattr(torchaudio.pipelines, bundle_name)
        self.encoder = bundle.get_model()
        for p in self.encoder.parameters():
            p.requires_grad = trainable
        enc_dim = bundle._params["encoder_embed_dim"]
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(enc_dim, num_classes))

    def forward(self, wave: torch.Tensor):
        with torch.no_grad():
            feats = self.encoder.extract_features(wave)[0]
        if isinstance(feats, (list, tuple)):
            feats = feats[-1] if isinstance(feats[-1], torch.Tensor) else feats[0]
        z = feats.mean(dim=1)  # [B, T, D] -> [B, D]
        return self.head(z)


def main():
    ap = argparse.ArgumentParser(description="Clasificación audio con wav2vec congelado + head lineal, split por participante.")
    ap.add_argument("--pkl", type=str, default="data/processed/multimodal_av_join_audio_cached.pkl")
    ap.add_argument("--audio-col", type=str, default="audio_cached_path")
    ap.add_argument("--label-col", type=str, default="action_proc")
    ap.add_argument("--participant-col", type=str, default="participant")
    ap.add_argument("--bundle", type=str, default="WAV2VEC2_BASE")
    ap.add_argument("--sr-cache", type=int, default=16000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_pickle(args.pkl).reset_index(drop=True)
    df_tr, df_val, df_te, info = split_by_participant(
        df, participant_col=args.participant_col, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    print(format_split_report(info))

    # Label remap if needed
    if df_tr[args.label_col].dtype == object:
        mapping = {v: i for i, v in enumerate(sorted(df[args.label_col].dropna().unique()))}
        for d in (df_tr, df_val, df_te):
            d[args.label_col] = d[args.label_col].map(mapping)

    num_classes = int(pd.Series(df_tr[args.label_col]).nunique())
    target_sr = getattr(torchaudio.pipelines, args.bundle).sample_rate

    train_ds = Wav2VecDataset(df_tr, args.audio_col, args.label_col, sr_cache=args.sr_cache, target_sr=target_sr)
    val_ds = Wav2VecDataset(df_val, args.audio_col, args.label_col, sr_cache=args.sr_cache, target_sr=target_sr)
    test_ds = Wav2VecDataset(df_te, args.audio_col, args.label_col, sr_cache=args.sr_cache, target_sr=target_sr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = Wav2VecClassifier(bundle_name=args.bundle, num_classes=num_classes, dropout=args.dropout, trainable=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def run_epoch(loader, train_flag: bool):
        if train_flag:
            model.train()
        else:
            model.eval()
        total_loss = 0.0
        ys, preds, logps = [], [], []
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            if train_flag:
                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            ys.append(y.detach().cpu())
            with torch.no_grad():
                logp = torch.log_softmax(logits, dim=1).cpu()
                logps.append(logp)
                preds.append(logp.argmax(dim=1))
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
        tr = run_epoch(train_loader, True)
        val = run_epoch(val_loader, False)
        history.append({"epoch": epoch, "train": tr, "val": val})
        print(f"Epoch {epoch:03d} | train acc={tr.get('acc',0):.3f} val acc={val.get('acc',0):.3f}")

    test_metrics = run_epoch(test_loader, False) if len(test_ds) else {}
    results_dir = Path("results")
    ensure_dir(results_dir)
    base_config = {
        "pkl": args.pkl,
        "audio_col": args.audio_col,
        "label_col": args.label_col,
        "participant_col": args.participant_col,
        "bundle": args.bundle,
        "sr_cache": args.sr_cache,
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
    save_metrics(all_metrics, results_dir, model_name="Wav2VecFreeze", config=base_config)
    hist_path = results_dir / "Wav2VecFreeze" / "history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    # Guardar split info
    info_path = results_dir / "Wav2VecFreeze" / "split_info.txt"
    info_path.write_text(format_split_report(info), encoding="utf-8")


if __name__ == "__main__":
    main()
