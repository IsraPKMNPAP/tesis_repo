from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure package root on path (same pattern as run_training.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loading.video_windows import (
    VideoWindowsDataset,
    adjust_paths_prefix,
    map_by_timestamp_or_order,
)
from src.models.video_torch import VideoCNNLSTM, train_gpu, extract_embeddings
from src.models.econ import fit_mnlogit
from utils.results_io import ensure_dir, default_prefix, save_text


def collate_windows(batch):
    # batch is a list of WindowSample. We build tensors and simple lists for meta.
    xs = []
    ys = []
    ts = []
    wids = []
    parts = []
    for b in batch:
        xs.append(b.x)
        ys.append(-1 if b.y is None else int(b.y))
        ts.append(b.timestamp)
        wids.append(b.window_id)
        parts.append(b.participant)

    # Stack along batch dimension; handle inputs of shape [T,C,H,W] or [C,H,W]
    if xs[0].dim() == 3:
        x = torch.stack(xs, dim=0)  # [B, C, H, W]
    elif xs[0].dim() == 4:
        x = torch.stack(xs, dim=0)  # [B, T, C, H, W]
    else:
        raise ValueError(f"Tensor de entrada con dimensión no soportada: {xs[0].shape}")

    y = torch.tensor(ys, dtype=torch.long)

    # Return a simple container compatible with attribute access in training code
    class B:
        def __init__(self, x, y, timestamp, window_id, participant):
            self.x = x
            self.y = y
            self.timestamp = timestamp
            self.window_id = window_id
            self.participant = participant

    return B(x=x, y=y, timestamp=ts, window_id=wids, participant=parts)


def main():
    ap = argparse.ArgumentParser(description="Train CNN+LSTM on precomputed video window tensors")
    ap.add_argument("--pickle", type=str, required=True, help="Ruta al X_proc_final.pkl con rutas y metadatos")
    ap.add_argument("--path-col", type=str, default="gpu_tensor_path", help="Columna con rutas a .pt")
    ap.add_argument("--label-col", type=str, default="action", help="Columna con label entero")
    ap.add_argument("--timestamp-col", type=str, default="timestamp")
    ap.add_argument("--window-id-col", type=str, default="window")
    ap.add_argument("--onedrive-prefix", type=str, default=None, help="Prefijo a reemplazar (Windows/OneDrive)")
    ap.add_argument("--linux-root", type=str, default=None, help="Raíz de datos en GPU/Linux")
    ap.add_argument("--map-by-timestamp", action="store_true", help="Mapear usando timestamp/orden a window_i.pt")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--cnn-emb", type=int, default=128)
    ap.add_argument("--lstm-hidden", type=int, default=128)
    ap.add_argument("--lstm-layers", type=int, default=1)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--num-classes", type=int, required=False, help="Si no se provee, se infiere de la columna de labels")
    ap.add_argument("--arkoudi", action="store_true", help="Usar cabeza Arkoudi (embeddings de clase, logits = z @ E^T)")
    ap.add_argument("--arkoudi-no-norm", action="store_true", help="Desactivar normalización L2 en Arkoudi head")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--prefix", type=str, default=None, help="Prefijo para resultados en results/")
    args = ap.parse_args()

    # Load dataframe from pickle
    pkl_path = Path(args.pickle)
    if not pkl_path.exists():
        raise FileNotFoundError(f"No existe el pickle: {pkl_path}")
    df = pd.read_pickle(pkl_path)

    # Optional path prefix adjustment
    if args.onedrive_prefix and args.linux_root:
        df = adjust_paths_prefix(df, column=args.path_col, src_prefix=args.onedrive_prefix, dst_prefix=args.linux_root)

    # Optional timestamp/order mapping to GPU window files
    if args.map_by_timestamp and args.linux_root:
        df = map_by_timestamp_or_order(df, timestamp_col=args.timestamp_col, gpu_root=args.linux_root)
        # Use the mapped column from here on
        path_col = "gpu_tensor_path"
    else:
        path_col = args.path_col

    # Split train/val
    idx = np.arange(len(df))
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    n_val = int(len(df) * args.val_split)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    ds_tr = VideoWindowsDataset(
        df_tr, path_col=path_col, label_col=args.label_col, timestamp_col=args.timestamp_col, window_id_col=args.window_id_col
    )
    ds_val = VideoWindowsDataset(
        df_val, path_col=path_col, label_col=args.label_col, timestamp_col=args.timestamp_col, window_id_col=args.window_id_col
    )

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_windows)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_windows)

    # Model
    model = VideoCNNLSTM(
        in_channels=ds_tr[0].x.shape[-3] if ds_tr[0].x.dim() >= 3 else 3,
        cnn_emb_dim=args.cnn_emb,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        bidirectional=args.bidirectional,
        num_classes=(args.num_classes if args.num_classes is not None else int(pd.Series(df[args.label_col]).nunique())),
        arkoudi=args.arkoudi,
        arkoudi_normalize=(not args.arkoudi_no_norm),
    )

    # Train
    model, hist = train_gpu(
        model=model,
        train_loader=dl_tr,
        val_loader=dl_val,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amp=True,
    )

    # Prepare results dir
    results_dir = Path("results")
    ensure_dir(results_dir)
    prefix = args.prefix or default_prefix(args.pickle, "videos")

    # Save model
    torch.save(model.state_dict(), results_dir / f"{prefix}_cnn_lstm.pt")
    # Save history
    pd.DataFrame({"loss": hist.losses, "acc": hist.accs}).to_csv(results_dir / f"{prefix}_history.csv", index=False)

    # Validation evaluation and probabilities
    from sklearn.metrics import classification_report
    model.eval()
    all_y_true, all_y_pred = [], []
    all_probs = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with torch.no_grad():
        for b in dl_val:
            x = b.x.to(device)
            y = b.y.to(device)
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred = logits.argmax(dim=1)
            all_y_true.extend(y.cpu().numpy().tolist())
            all_y_pred.extend(pred.cpu().numpy().tolist())
            all_probs.append(probs)
    if all_probs:
        probs = np.concatenate(all_probs, axis=0)
        report = classification_report(all_y_true, all_y_pred)
        print("\n=== Validation (Torch CNN+LSTM) ===")
        print(report)
        # Save report and probabilities
        save_text(report, results_dir / f"{prefix}_val_report.txt")
        prob_df = pd.DataFrame(probs, columns=[f"class_{i}" for i in range(probs.shape[1])])
        prob_df.to_csv(results_dir / f"{prefix}_val_proba.csv", index=False)

    # Extract embeddings on full dataset (train+val)
    ds_all = VideoWindowsDataset(
        df.reset_index(drop=True), path_col=path_col, label_col=args.label_col, timestamp_col=args.timestamp_col, window_id_col=args.window_id_col
    )
    dl_all = DataLoader(ds_all, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_windows)
    embs, labels, meta = extract_embeddings(model, dl_all)
    emb_df = pd.DataFrame(embs)
    emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]
    emb_df["label"] = labels
    emb_df["timestamp"] = [m[0] for m in meta]
    emb_df["window_id"] = [m[1] for m in meta]
    emb_df["participant"] = [m[2] for m in meta]
    emb_csv = results_dir / f"{prefix}_embeddings.csv"
    emb_df.to_csv(emb_csv, index=False)

    # Econometric analysis: MNLogit on embeddings (only rows with labels)
    labeled_mask = pd.notna(emb_df["label"]).values
    if labeled_mask.sum() > 0:
        X_proc = emb_df.loc[labeled_mask, [c for c in emb_df.columns if c.startswith("emb_")]].to_numpy(dtype=float)
        y_enc = emb_df.loc[labeled_mask, "label"].astype(int).to_numpy()
        try:
            res = fit_mnlogit(X_proc, y_enc)
            summary = res.summary().as_text()
            print("\n=== MNLogit (statsmodels) on embeddings ===")
            print(summary)
            save_text(summary, results_dir / f"{prefix}_mnlogit_embeddings_summary.txt")
        except Exception as e:
            save_text(f"MNLogit fallo: {e}", results_dir / f"{prefix}_mnlogit_embeddings_error.txt")

    # Save run config
    config = {
        "pickle": args.pickle,
        "path_col": args.path_col,
        "label_col": args.label_col,
        "timestamp_col": args.timestamp_col,
        "window_id_col": args.window_id_col,
        "onedrive_prefix": args.onedrive_prefix,
        "linux_root": args.linux_root,
        "map_by_timestamp": args.map_by_timestamp,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "cnn_emb": args.cnn_emb,
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "bidirectional": args.bidirectional,
        "num_classes": args.num_classes,
        "val_split": args.val_split,
        "prefix": prefix,
    }
    (results_dir / f"{prefix}_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
