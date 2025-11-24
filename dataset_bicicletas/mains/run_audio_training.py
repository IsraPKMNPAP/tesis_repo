#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import WeightedRandomSampler, DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_loading.audio_windows import (
    split_by_participant,
    create_audio_dataloaders,
    AudioSegmentDataset,
)
from src.models.audio_cnn import AudioCNNLogit
from src.models.audio_tcn import AudioTCNLogit
from src.models.audio_wav2vec import AudioWav2VecLogit, get_bundle
from utils.results_io import (
    ensure_dir,
    compute_run_hash,
    artifact_name,
    save_text,
    save_probs,
    register_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline CNN/logit sobre ventanas de audio crudo.")
    parser.add_argument("--pickle", required=True, help="Ruta a X_vid_aud.pkl")
    parser.add_argument("--audio-root", required=True, help="Carpeta con raw_audio_PXX.wav")
    parser.add_argument("--participant-col", default="participant")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--start-col", default="audio_segment_start")
    parser.add_argument("--label-col", default="action_proc")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--arch", choices=["cnn", "tcn", "wav2vec"], default="cnn")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--participant-prefix", default="P")
    parser.add_argument("--participant-zero-pad", type=int, default=2)
    parser.add_argument("--filename-template", default="raw_audio_{participant}.wav")
    parser.add_argument("--cache-audio", action="store_true", help="Cachear audio completo por participante en RAM para acelerar slicing.")
    parser.add_argument("--aug-noise-std", type=float, default=0.0, help="Desvío estándar del ruido gaussiano en waveform.")
    parser.add_argument("--aug-prob", type=float, default=0.0, help="Probabilidad de aplicar ruido gaussiano.")
    parser.add_argument("--cnn-channels", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tcn-channels", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--tcn-kernel-size", type=int, default=3)
    parser.add_argument("--tcn-dropout", type=float, default=0.2)
    parser.add_argument("--wav2vec-bundle", default="WAV2VEC2_BASE")
    parser.add_argument("--wav2vec-trainable", action="store_true")
    parser.add_argument("--wav2vec-dropout", type=float, default=0.1)
    parser.add_argument("--results-prefix", default="AudioCNN")
    parser.add_argument("--class-weighted", action="store_true")
    parser.add_argument("--balance-sampler", action="store_true", help="Usar WeightedRandomSampler para balancear clases.")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing para CrossEntropy.")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=5, help="Cortesía: detener si val_acc no mejora en este número de épocas.")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.1, help="Mejora mínima en val_acc para resetear paciencia.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> Tuple[float, float]:
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0
    for batch in loader:
        waveforms = batch["waveform"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(waveforms)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        preds = logits.argmax(dim=1)
        running_loss += loss.item() * labels.size(0)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / max(total, 1), running_correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int], np.ndarray]:
    model.eval()
    running_loss, running_correct, total = 0.0, 0, 0
    all_true: List[int] = []
    all_pred: List[int] = []
    all_probs: List[np.ndarray] = []
    for batch in loader:
        waveforms = batch["waveform"].to(device)
        labels = batch["label"].to(device)
        logits = model(waveforms)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        running_loss += loss.item() * labels.size(0)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())
        all_probs.append(probs.cpu().numpy())
    if all_probs:
        prob_matrix = np.concatenate(all_probs, axis=0)
    else:
        out_dim = getattr(getattr(model, "classifier", None), "out_features", 0)
        prob_matrix = np.empty((0, out_dim))
    return running_loss / max(total, 1), running_correct / max(total, 1), all_true, all_pred, prob_matrix


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        args.device if args.device not in {"auto", "cuda"} else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Ajustar sample_rate si se usa wav2vec (debe coincidir con el bundle)
    if args.arch == "wav2vec":
        bundle = get_bundle(args.wav2vec_bundle)
        args.sample_rate = bundle.sample_rate

    df = pd.read_pickle(args.pickle)
    df = df.dropna(subset=[args.participant_col, args.start_col, args.label_col]).reset_index(drop=True)
    df[args.start_col] = df[args.start_col].astype(float)

    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df[args.label_col].astype(str))
    num_classes = len(label_encoder.classes_)
    train_df, val_df = split_by_participant(df, args.participant_col, args.val_split, args.seed)

    dataset_kwargs: Dict[str, object] = dict(
        audio_root=args.audio_root,
        participant_col=args.participant_col,
        start_col=args.start_col,
        label_col="label_id",
        timestamp_col=args.timestamp_col,
        window_seconds=args.window_seconds,
        sample_rate=args.sample_rate,
        filename_template=args.filename_template,
        participant_prefix=args.participant_prefix,
        participant_zero_pad=args.participant_zero_pad,
        strict=True,
        cache_audio=args.cache_audio,
        aug_noise_std=args.aug_noise_std,
        aug_prob=args.aug_prob,
    )
    # Sampler balanceado opcional
    if args.balance_sampler:
        class_sample_counts = np.bincount(train_df["label_id"], minlength=num_classes)
        weights = 1.0 / np.maximum(class_sample_counts, 1)
        sample_weights = weights[train_df["label_id"].to_numpy()]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            AudioSegmentDataset(train_df, **dataset_kwargs),
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
    else:
        train_loader, val_loader = create_audio_dataloaders(
            train_df,
            val_df,
            dataset_kwargs=dataset_kwargs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    # Asegurar val_loader si sampler se usó
    if args.balance_sampler:
        val_loader = DataLoader(
            AudioSegmentDataset(val_df, **dataset_kwargs),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

    if args.arch == "cnn":
        model = AudioCNNLogit(
            sample_rate=args.sample_rate,
            num_classes=num_classes,
            n_mels=args.n_mels,
            cnn_channels=args.cnn_channels,
            dropout=args.dropout,
        )
    elif args.arch == "tcn":
        model = AudioTCNLogit(
            sample_rate=args.sample_rate,
            num_classes=num_classes,
            n_mels=args.n_mels,
            tcn_channels=args.tcn_channels,
            kernel_size=args.tcn_kernel_size,
            dropout=args.tcn_dropout,
        )
    elif args.arch == "wav2vec":
        model = AudioWav2VecLogit(
            bundle_name=args.wav2vec_bundle,
            num_classes=num_classes,
            trainable=args.wav2vec_trainable,
            dropout=args.wav2vec_dropout,
        )
    else:
        raise ValueError(f"Arquitectura no soportada: {args.arch}")
    model = model.to(device)

    class_weights = None
    if args.class_weighted:
        counts = np.bincount(train_df["label_id"], minlength=num_classes)
        weights = (counts.sum() / np.maximum(counts, 1)).astype(np.float32)
        class_weights = torch.tensor(weights / weights.mean(), device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_rows: List[Dict[str, float]] = []
    best_state = None
    best_val_acc = 0.0
    no_improve_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, args.grad_clip)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        history_rows.append(
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
        )
        print(
            f"[{epoch:03d}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )
        if val_acc >= best_val_acc + args.early_stop_min_delta:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= args.early_stop_patience:
            print(
                f"[early-stop] Sin mejora >= {args.early_stop_min_delta:.3f} en val_acc por "
                f"{args.early_stop_patience} épocas. Se detiene en la época {epoch}."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_acc, y_true, y_pred, y_prob = evaluate(model, val_loader, criterion, device)
    y_true_labels = label_encoder.inverse_transform(y_true)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    report = classification_report(y_true_labels, y_pred_labels, labels=label_encoder.classes_)
    print("\n=== Reporte validación ===")
    print(report)

    results_dir = Path("results")
    ensure_dir(results_dir)
    config = vars(args)
    config["num_classes"] = num_classes
    config["device_used"] = str(device)
    model_name = args.results_prefix
    run_hash = compute_run_hash(config, sys.argv, model=model_name)

    (results_dir / artifact_name(model_name, "config", run_hash, "json")).write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "label_encoder": label_encoder.classes_.tolist(),
            "config": config,
        },
        results_dir / artifact_name(model_name, "model", run_hash, "pt"),
    )
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(results_dir / artifact_name(model_name, "history", run_hash, "csv"), index=False)
    save_text(report, results_dir / artifact_name(model_name, "eval_report", run_hash, "txt"))
    save_probs(
        probs=y_prob,
        classes=label_encoder.classes_,
        out_path=results_dir / artifact_name(model_name, "eval_proba", run_hash, "csv"),
    )
    register_run(results_dir, run_hash, model_name, cmd=" ".join(sys.argv), config=config)


if __name__ == "__main__":
    main()
