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
from src.models.video_backbone_lstm import FrameBackboneLSTM
from src.models.video_torch import train_gpu, extract_embeddings
from src.models.econ import fit_mnlogit
from utils.results_io import ensure_dir, save_text, compute_run_hash, artifact_name, register_run


def collate_windows(batch):
    xs, ys, ts, wids, parts = [], [], [], [], []
    default_map = {
        'accelerate': 0,
        'brake': 1,
        'decelerate': 2,
        'maintain speed': 3,
        'wait': 4,
    }
    def _coerce_label(v):
        if v is None:
            return -1
        try:
            if isinstance(v, str):
                return int(default_map.get(v, -1))
            if isinstance(v, torch.Tensor):
                return int(v.item())
            return int(v)
        except Exception:
            return -1
    for b in batch:
        xs.append(b.x)
        ys.append(_coerce_label(b.y))
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
    ap = argparse.ArgumentParser(description="Train CLIP/ViT + LSTM models on linked video windows")
    ap.add_argument("--pickle", type=str, required=True, help="Pickle procesado con columna gpu_tensor_path")
    ap.add_argument("--path-col", type=str, default="gpu_tensor_path")
    ap.add_argument("--label-col", type=str, default="action")
    ap.add_argument("--timestamp-col", type=str, default="timestamp")
    ap.add_argument("--window-id-col", type=str, default="window")

    # Backbone config
    ap.add_argument("--backbone", type=str, default="vit", choices=["vit", "clip"], help="Tipo de backbone de frames")
    ap.add_argument("--backbone-name", type=str, default="vit_b_16", help="Nombre de modelo (vit_b_16 o ViT-B-16 para CLIP)")
    ap.add_argument("--backbone-trainable", action="store_true", help="Habilitar fine-tuning del backbone (por defecto congelado)")
    ap.add_argument("--backbone-unfreeze-last", type=int, default=0, help="Número de bloques finales a descongelar para fine-tuning")
    ap.add_argument("--target-size", type=int, default=224, help="Tamaño de entrada para el backbone")

    # LSTM + head
    ap.add_argument("--lstm-hidden", type=int, default=256)
    ap.add_argument("--lstm-layers", type=int, default=1)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--num-classes", type=int, required=False)
    ap.add_argument("--arkoudi", action="store_true")
    ap.add_argument("--arkoudi-no-norm", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.1)

    # Train
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--class-weighted", action="store_true")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--scheduler", type=str, default=None, choices=[None, "step", "cosine", "plateau"], help="Scheduler de LR")
    ap.add_argument("--step-size", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--t-max", type=int, default=None)
    ap.add_argument("--plateau-patience", type=int, default=3)
    ap.add_argument("--plateau-factor", type=float, default=0.5)
    ap.add_argument("--debug-eval", action="store_true")
    args = ap.parse_args()

    pkl = Path(os.path.expanduser(args.pickle))
    if not pkl.exists():
        raise FileNotFoundError(f"No existe el pickle: {pkl}")
    df = pd.read_pickle(pkl)
    if args.path_col not in df.columns:
        raise KeyError(f"'{args.path_col}' no está en el DataFrame. Ejecuta primero run_link_video_tensors.")

    # Split
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
        prefer_df_label=True,
    )
    ds_val = VideoWindowsDataset(
        df_val,
        path_col=args.path_col,
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        window_id_col=args.window_id_col,
        prefer_df_label=True,
    )

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_windows)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_windows)

    model = FrameBackboneLSTM(
        backbone=args.backbone,
        backbone_name=args.backbone_name,
        backbone_trainable=args.backbone_trainable,
        backbone_unfreeze_last=args.backbone_unfreeze_last,
        target_size=args.target_size,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        bidirectional=args.bidirectional,
        num_classes=(args.num_classes if args.num_classes is not None else int(pd.Series(df[args.label_col]).nunique())),
        arkoudi=args.arkoudi,
        arkoudi_normalize=(not args.arkoudi_no_norm),
        dropout=args.dropout,
    )

    # Class weights
    class_weights = None
    if args.class_weighted:
        y_series = pd.to_numeric(df_tr[args.label_col], errors='coerce').dropna().astype(int)
        k = int(pd.Series(df[args.label_col]).nunique())
        counts = y_series.value_counts().reindex(range(k), fill_value=0)
        freq = counts.values.astype(float)
        inv = np.where(freq > 0, 1.0 / freq, 0.0)
        if inv.sum() > 0:
            inv = inv * (len(inv) / max(1.0, inv.sum()))
        class_weights = torch.tensor(inv, dtype=torch.float32)

    # Scheduler kwargs
    sched_kwargs = {}
    if args.scheduler == "step":
        sched_kwargs = {"step_size": args.step_size, "gamma": args.gamma}
    elif args.scheduler == "cosine":
        if args.t_max is not None:
            sched_kwargs = {"t_max": args.t_max}
    elif args.scheduler == "plateau":
        sched_kwargs = {"patience": args.plateau_patience, "factor": args.plateau_factor}

    model, hist = train_gpu(
        model=model,
        train_loader=dl_tr,
        val_loader=dl_val,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amp=True,
        class_weights=class_weights,
        scheduler=args.scheduler,
        scheduler_kwargs=sched_kwargs,
        grad_clip=1.0,
        label_smoothing=0.05,
    )

    # Results naming
    results_dir = Path("results")
    ensure_dir(results_dir)
    model_name = f"{args.backbone.upper()}-LSTM"
    config = {
        "pickle": str(pkl),
        "path_col": args.path_col,
        "label_col": args.label_col,
        "timestamp_col": args.timestamp_col,
        "window_id_col": args.window_id_col,
        "backbone": args.backbone,
        "backbone_name": args.backbone_name,
        "backbone_trainable": args.backbone_trainable,
        "target_size": args.target_size,
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "bidirectional": args.bidirectional,
        "num_classes": (args.num_classes if args.num_classes is not None else int(pd.Series(df[args.label_col]).nunique())),
        "arkoudi": args.arkoudi,
        "arkoudi_no_norm": args.arkoudi_no_norm,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "class_weighted": args.class_weighted,
        "scheduler": args.scheduler,
    }
    run_hash = compute_run_hash(config, sys.argv, model=model_name)

    # Validation
    all_y_true, all_y_pred, all_probs, all_logits = [], [], [], []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
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
            all_logits.append(logits.detach().cpu().numpy())

    # Save artifacts
    torch.save(model.state_dict(), results_dir / artifact_name(model_name, "model", run_hash, "pt"))
    pd.DataFrame({"loss": hist.losses, "acc": hist.accs}).to_csv(
        results_dir / artifact_name(model_name, "history", run_hash, "csv"), index=False
    )
    if all_probs:
        probs = np.concatenate(all_probs, axis=0)
        report = classification_report(all_y_true, all_y_pred, zero_division=0)
        print("\n=== Validation (Backbone + LSTM) ===")
        print(report)
        save_text(report, results_dir / artifact_name(model_name, "eval_report", run_hash, "txt"))
        pd.DataFrame(probs, columns=[f"class_{i}" for i in range(probs.shape[1])]).to_csv(
            results_dir / artifact_name(model_name, "eval_proba", run_hash, "csv"), index=False
        )

    # Embeddings + MNLogit
    ds_all = VideoWindowsDataset(
        df.reset_index(drop=True),
        path_col=args.path_col,
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        window_id_col=args.window_id_col,
        prefer_df_label=True,
    )
    dl_all = DataLoader(ds_all, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_windows)
    from src.models.video_torch import extract_embeddings as extract_embs
    embs, labels, meta = extract_embs(model, dl_all)
    emb_df = pd.DataFrame(embs)
    emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]
    emb_df["label"] = labels
    emb_df["timestamp"] = [m[0] for m in meta]
    emb_df["window_id"] = [m[1] for m in meta]
    emb_df["participant"] = [m[2] for m in meta]
    emb_df.to_csv(results_dir / artifact_name(model_name, "embeddings", run_hash, "csv"), index=False)

    labeled_mask = pd.notna(emb_df["label"]).values
    if labeled_mask.sum() > 0:
        X_proc = emb_df.loc[labeled_mask, [c for c in emb_df.columns if c.startswith("emb_")]].to_numpy(dtype=float)
        y_enc = emb_df.loc[labeled_mask, "label"].astype(int).to_numpy()
        try:
            res = fit_mnlogit(X_proc, y_enc)
            summary = res.summary().as_text()
            print("\n=== MNLogit (statsmodels) on embeddings ===")
            print(summary)
            save_text(summary, results_dir / artifact_name(model_name, "mnlogit_summary", run_hash, "txt"))
        except Exception as e:
            save_text(f"MNLogit fallo: {e}", results_dir / artifact_name(model_name, "mnlogit_error", run_hash, "txt"))

    # Save config + index
    (results_dir / artifact_name(model_name, "config", run_hash, "json")).write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    register_run(results_dir, run_hash, model_name, cmd=" ".join(sys.argv), config=config)


if __name__ == "__main__":
    main()
