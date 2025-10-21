from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Ensure package root on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loading.video_windows import VideoWindowsDataset
from src.models.video_torch import VideoCNNLSTM, train_gpu, extract_embeddings
from src.models.econ import fit_mnlogit
from utils.results_io import ensure_dir, default_prefix, save_text


def collate_windows(batch):
    xs, ys, ts, wids, parts = [], [], [], [], []
    for b in batch:
        xs.append(b.x)
        ys.append(-1 if b.y is None else int(b.y))
        ts.append(b.timestamp)
        wids.append(b.window_id)
        parts.append(b.participant)

    if xs[0].dim() == 3:
        x = torch.stack(xs, dim=0)
    elif xs[0].dim() == 4:
        x = torch.stack(xs, dim=0)
    else:
        raise ValueError(f"Tensor de entrada con dimensión no soportada: {xs[0].shape}")

    y = torch.tensor(ys, dtype=torch.long)

    class B:
        def __init__(self, x, y, timestamp, window_id, participant):
            self.x = x
            self.y = y
            self.timestamp = timestamp
            self.window_id = window_id
            self.participant = participant

    return B(x=x, y=y, timestamp=ts, window_id=wids, participant=parts)


def main():
    ap = argparse.ArgumentParser(description="Train CNN+LSTM on linked video window tensors (processed pickle)")
    ap.add_argument(
        "--pickle",
        type=str,
        default="~/projects/tesis_repo/dataset_bicicletas/data/processed/X_proc_final_linked.pkl",
        help="Ruta al pickle procesado con columna 'gpu_tensor_path'",
    )
    ap.add_argument("--path-col", type=str, default="gpu_tensor_path", help="Columna con rutas a .pt")
    ap.add_argument("--label-col", type=str, default="action", help="Columna con label entero")
    ap.add_argument("--prefer-df-label", action="store_true", help="Forzar uso de la columna de labels del DataFrame incluso si el .pt trae 'label'")
    ap.add_argument("--timestamp-col", type=str, default="timestamp")
    ap.add_argument("--window-id-col", type=str, default="window")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--cnn-emb", type=int, default=128)
    ap.add_argument("--lstm-hidden", type=int, default=128)
    ap.add_argument("--lstm-layers", type=int, default=1)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--num-classes", type=int, required=False)
    ap.add_argument("--arkoudi", action="store_true")
    ap.add_argument("--arkoudi-no-norm", action="store_true")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--prefix", type=str, default=None)
    args = ap.parse_args()

    pkl_path = Path(os.path.expanduser(args.pickle))
    if not pkl_path.exists():
        raise FileNotFoundError(f"No existe el pickle: {pkl_path}")
    df = pd.read_pickle(pkl_path)
    # Verify path column exists
    if args.path_col not in df.columns:
        # Try a common fallback
        if "gpu_tensor_path" in df.columns:
            print(f"Aviso: columna '{args.path_col}' no existe. Usando 'gpu_tensor_path'.")
            args.path_col = "gpu_tensor_path"
        else:
            raise KeyError(
                f"El pickle no contiene la columna de rutas '{args.path_col}'. Ejecuta primero run_link_video_tensors para generar 'gpu_tensor_path'."
            )

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
        df_tr,
        path_col=args.path_col,
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        window_id_col=args.window_id_col,
        prefer_df_label=args.prefer_df_label,
    )
    ds_val = VideoWindowsDataset(
        df_val,
        path_col=args.path_col,
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        window_id_col=args.window_id_col,
        prefer_df_label=args.prefer_df_label,
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

    # Results
    results_dir = Path("results")
    ensure_dir(results_dir)
    prefix = args.prefix or default_prefix(pkl_path, "videos")

    torch.save(model.state_dict(), results_dir / f"{prefix}_cnn_lstm.pt")
    pd.DataFrame({"loss": hist.losses, "acc": hist.accs}).to_csv(results_dir / f"{prefix}_history.csv", index=False)

    # Validation
    model.eval()
    all_y_true, all_y_pred, all_probs = [], [], []
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
        save_text(report, results_dir / f"{prefix}_val_report.txt")
        pd.DataFrame(probs, columns=[f"class_{i}" for i in range(probs.shape[1])]).to_csv(
            results_dir / f"{prefix}_val_proba.csv", index=False
        )

    # Embeddings + MNLogit
    ds_all = VideoWindowsDataset(
        df.reset_index(drop=True),
        path_col=args.path_col,
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        window_id_col=args.window_id_col,
        prefer_df_label=args.prefer_df_label,
    )
    dl_all = DataLoader(ds_all, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_windows)
    embs, labels, meta = extract_embeddings(model, dl_all)
    emb_df = pd.DataFrame(embs)
    emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]
    emb_df["label"] = labels
    emb_df["timestamp"] = [m[0] for m in meta]
    emb_df["window_id"] = [m[1] for m in meta]
    emb_df["participant"] = [m[2] for m in meta]
    emb_df.to_csv(results_dir / f"{prefix}_embeddings.csv", index=False)

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
        "pickle": str(pkl_path),
        "path_col": args.path_col,
        "label_col": args.label_col,
        "timestamp_col": args.timestamp_col,
        "window_id_col": args.window_id_col,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "cnn_emb": args.cnn_emb,
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "bidirectional": args.bidirectional,
        "num_classes": (args.num_classes if args.num_classes is not None else int(pd.Series(df[args.label_col]).nunique())),
        "arkoudi": args.arkoudi,
        "arkoudi_no_norm": args.arkoudi_no_norm,
        "val_split": args.val_split,
        "prefix": prefix,
    }
    (results_dir / f"{prefix}_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
