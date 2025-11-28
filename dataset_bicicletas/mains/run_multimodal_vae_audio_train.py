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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler

# Ensure package root on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loading.multimodal_audio import MultimodalAudioDataset, collate_multimodal_audio
from src.models.mm_vae_audio import DeterministicMMVAEAudio, VariationalMMVAEAudio
from utils.results_io import (
    ensure_dir,
    save_text,
    save_model_pickle,
    compute_run_hash,
    artifact_name,
    register_run,
)
from utils.features import load_features_file
from src.data_cleaning.cleaning import categorias_a_str, convertir_a_categorico


def _to_float_tensor(mat):
    import numpy as np
    try:
        arr = mat.toarray()
    except Exception:
        try:
            arr = np.asarray(mat)
        except Exception:
            arr = mat
    return torch.tensor(arr.astype(np.float32), dtype=torch.float32)


def split_train_val(df: pd.DataFrame, label_col: str, val_split: float = 0.2, seed: int = 42):
    rng = np.random.RandomState(seed)
    if val_split <= 0 or val_split >= 1:
        return df.reset_index(drop=True), df.iloc[0:0].copy()
    y = pd.to_numeric(df[label_col], errors="coerce") if df[label_col].dtype != object else df[label_col]
    if y.notna().all():
        labels = y
        uniq = pd.Series(labels).unique()
        val_idx = []
        for c in uniq:
            idx = np.where(labels == c)[0]
            k = int(max(1, round(len(idx) * val_split)))
            val_idx.extend(rng.choice(idx, size=min(k, len(idx)), replace=False))
        val_idx = sorted(set(val_idx))
    else:
        n = len(df)
        k = int(round(n * val_split))
        val_idx = sorted(rng.choice(np.arange(n), size=k, replace=False).tolist())
    mask = np.zeros(len(df), dtype=bool)
    mask[val_idx] = True
    df_val = df.iloc[mask].reset_index(drop=True)
    df_tr = df.iloc[~mask].reset_index(drop=True)
    return df_tr, df_val


def main():
    ap = argparse.ArgumentParser(description="Train multimodal VAE (tabular + video + audio) end-to-end")
    ap.add_argument("--pkl", type=str, default="data/processed/multimodal_join.pkl", help="Ruta al pickle multimodal")
    ap.add_argument("--label-col", type=str, default="action_proc")
    ap.add_argument("--features", nargs="*", default=None, help="Columnas tabulares a usar")
    ap.add_argument("--features-file", type=str, default="utils/feature_sets/exp1.json", help="Archivo con lista de features")
    ap.add_argument("--path-col", type=str, default="gpu_tensor_path")
    ap.add_argument("--video-root", type=str, default=None, help="Raíz para prefijar paths de video si son relativos (ej: /mnt/.../video_tensors)")
    ap.add_argument("--audio-col", type=str, default="audio_path", help="Columna con ruta directa al audio (opcional)")
    ap.add_argument("--timestamp-col", type=str, default="timestamp")
    ap.add_argument("--window-id-col", type=str, default="window")
    ap.add_argument("--participant-col", type=str, default="participant")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--deterministic", action="store_true", help="Usar VAE determinista (por defecto, variacional)")
    ap.add_argument("--tab-emb", type=int, default=128)
    ap.add_argument("--shared-dim", type=int, default=64)
    ap.add_argument("--audio-emb", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--w-rec-tab", type=float, default=1.0)
    ap.add_argument("--w-rec-vid", type=float, default=1.0)
    ap.add_argument("--w-cls", type=float, default=1.0)
    ap.add_argument("--w-kl", type=float, default=1.0)
    ap.add_argument("--kl-anneal-steps", type=int, default=1000)
    ap.add_argument("--save-embeddings", action="store_true")
    # Regularización/opt
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--scheduler", type=str, default="none", choices=["none", "step", "cosine", "plateau"], help="Scheduler de LR")
    ap.add_argument("--step-size", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--t-max", type=int, default=None)
    ap.add_argument("--plateau-patience", type=int, default=3)
    ap.add_argument("--plateau-factor", type=float, default=0.5)
    ap.add_argument("--early-stop-patience", type=int, default=6)
    ap.add_argument("--early-stop-delta", type=float, default=0.01)
    # Video backbone fine-tuning
    ap.add_argument("--video-backbone", type=str, default="vit", choices=["vit", "clip"])
    ap.add_argument("--video-name", type=str, default="vit_b_16")
    ap.add_argument("--video-trainable", action="store_true", default=True)
    ap.add_argument("--video-unfreeze-last", type=int, default=1)
    ap.add_argument("--video-target-size", type=int, default=224)
    ap.add_argument("--video-lstm-hidden", type=int, default=256)
    ap.add_argument("--video-lstm-layers", type=int, default=1)
    ap.add_argument("--video-bidirectional", action="store_true")
    ap.add_argument("--video-dropout", type=float, default=0.0)
    # Fusion mode
    ap.add_argument("--fusion", type=str, default="early", choices=["early", "late"])
    ap.add_argument("--late-alpha", type=float, default=0.5)
    # Alignment/contrastive
    ap.add_argument("--w-align", type=float, default=0.0)
    ap.add_argument("--w-contrastive", type=float, default=0.0)
    ap.add_argument("--contrastive-temp", type=float, default=0.07)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--modality-dropout", type=float, default=0.0)
    ap.add_argument("--fuse-dropout", type=float, default=0.0)
    ap.add_argument("--w-aux-tab", type=float, default=0.1, help="Peso de CE auxiliar sobre logits_tab")
    ap.add_argument("--w-aux-vid", type=float, default=0.1, help="Peso de CE auxiliar sobre logits_vid")
    ap.add_argument("--w-aux-aud", type=float, default=0.05, help="Peso de CE auxiliar sobre logits_aud")
    # Modality toggles
    ap.add_argument("--use-tabular", action="store_true")
    ap.add_argument("--use-video", action="store_true")
    ap.add_argument("--use-audio", action="store_true")
    # Overfit mode
    ap.add_argument("--overfit-batches", type=int, default=0)
    # Class weighting
    ap.add_argument("--class-weighted", action="store_true", default=True)
    # Warmup
    ap.add_argument("--warmup-epochs", type=int, default=0)
    ap.add_argument("--warmup-modality", type=str, default="both", choices=["both", "tabular", "video", "audio"])
    ap.add_argument("--warmup-disable-contrastive", action="store_true")
    # Normalización
    ap.add_argument("--tabular-scaler", type=str, default="robust", choices=["standard", "robust"])
    ap.add_argument("--video-norm", type=str, default="imagenet", choices=["imagenet", "per_channel", "none"])
    ap.add_argument("--audio-sr", type=int, default=16000)
    ap.add_argument("--audio-duration", type=float, default=5.0)
    ap.add_argument("--audio-norm", type=str, default="per_channel", choices=["per_channel", "none"])
    ap.add_argument(
        "--audio-root",
        type=str,
        default="/mnt/otra_particion/home/israel_gpu_data/audio_data_raw/audio_participantes_validos",
        help="Raíz donde viven los raw_audio_<PARTICIPANTE>.wav",
    )
    ap.add_argument("--audio-template", type=str, default="raw_audio_{participant}.wav")
    ap.add_argument("--audio-fallback-template", type=str, default=None)
    ap.add_argument("--audio-start-col", type=str, default="audio_segment_start")
    ap.add_argument("--debug-batch", action="store_true", help="Imprime shapes/min-max del primer batch y sale")
    args = ap.parse_args()
    if args.scheduler == "none":
        args.scheduler = None

    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        raise FileNotFoundError(f"No existe el pickle multimodal: {pkl_path}")
    df = pd.read_pickle(pkl_path).reset_index(drop=True)
    # Si se provee video_root, prefijar si las rutas no son absolutas
    if args.video_root:
        root = Path(args.video_root)
        df[args.path_col] = df[args.path_col].astype(str).apply(
            lambda p: str(root / p) if not Path(p).is_absolute() else p
        )
    has_audio = bool(args.audio_root)
    if not has_audio:
        print("[WARN] No se proporcionó audio_root; se desactiva audio.")

    # Features
    tab_cols = args.features
    if args.features_file:
        loaded = load_features_file(args.features_file)
        if loaded:
            tab_cols = loaded
    if not tab_cols:
        drop_cols = {args.path_col, args.audio_col, args.label_col, args.timestamp_col, args.window_id_col, "participant", "session_id"}
        tab_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    tab_cols = [c for c in tab_cols if c in df.columns]

    # Label mapping
    default_class_map = {
        "accelerate": 0,
        "brake": 1,
        "decelerate": 2,
        "maintain speed": 3,
        "wait": 4,
    }
    if args.label_col not in df.columns:
        raise KeyError(f"Falta la columna de etiqueta '{args.label_col}' en el pickle multimodal")
    if df[args.label_col].dtype == object:
        df[args.label_col] = df[args.label_col].map(default_class_map)
    num_classes = int(pd.Series(df[args.label_col]).nunique())

    # Split
    df_tr, df_val = split_train_val(df, label_col=args.label_col, val_split=args.val_split)

    # Preprocesamiento tabular
    X_tr_raw = df_tr[tab_cols].copy()
    X_val_raw = df_val[tab_cols].copy()
    X_tr_prep = convertir_a_categorico(categorias_a_str(X_tr_raw))
    X_val_prep = convertir_a_categorico(categorias_a_str(X_val_raw))
    numeric = X_tr_prep.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical = X_tr_prep.select_dtypes(include=["category"]).columns.tolist()
    scaler_cls = RobustScaler if args.tabular_scaler == "robust" else StandardScaler
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler_cls(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    X_tr_mat = preprocessor.fit_transform(X_tr_prep)
    X_val_mat = preprocessor.transform(X_val_prep)

    results_dir = Path("results")
    ensure_dir(results_dir)
    tmp_cfg = {"pkl": str(pkl_path), "features": tab_cols, "label_col": args.label_col}
    pre_hash = compute_run_hash(tmp_cfg, sys.argv, model="MMVAEAudio_Preproc")
    save_model_pickle(preprocessor, results_dir / artifact_name("MMVAEAudio", "preprocessor", pre_hash, "pkl"))

    # Video normalization transform
    def _video_transform(x: torch.Tensor) -> torch.Tensor:
        if args.video_norm == "none":
            return x
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if args.video_norm == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
            if x.size(1) == 3:
                x = (x - mean) / std
            return x if x.size(0) > 1 else x.squeeze(0)
        if args.video_norm == "per_channel":
            eps = 1e-6
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            std = x.std(dim=(0, 2, 3), keepdim=True)
            x = (x - mean) / (std + eps)
            return x if x.size(0) > 1 else x.squeeze(0)
        return x

    # Datasets / loaders
    ds_tr = MultimodalAudioDataset(
        df_tr,
        tab_columns=tab_cols,
        X_tab_array=_to_float_tensor(X_tr_mat),
        path_col=args.path_col,
        participant_col=args.participant_col,
        audio_start_col=args.audio_start_col,
        audio_root=args.audio_root if has_audio else None,
        audio_template=args.audio_template,
        audio_fallback_template=args.audio_fallback_template,
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        window_id_col=args.window_id_col,
        prefer_df_label=True,
        class_map=default_class_map,
        video_transform=_video_transform,
        audio_sr=args.audio_sr,
        audio_duration=args.audio_duration,
        audio_norm=args.audio_norm,
    )
    ds_val = MultimodalAudioDataset(
        df_val,
        tab_columns=tab_cols,
        X_tab_array=_to_float_tensor(X_val_mat),
        path_col=args.path_col,
        participant_col=args.participant_col,
        audio_start_col=args.audio_start_col,
        audio_root=args.audio_root if has_audio else None,
        audio_template=args.audio_template,
        audio_fallback_template=args.audio_fallback_template,
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        window_id_col=args.window_id_col,
        prefer_df_label=True,
        class_map=default_class_map,
        video_transform=_video_transform,
        audio_sr=args.audio_sr,
        audio_duration=args.audio_duration,
        audio_norm=args.audio_norm,
    )

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_multimodal_audio)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_multimodal_audio)

    # Modality defaults
    use_tab_default = args.use_tabular or (not args.use_tabular and not args.use_video and not args.use_audio)
    use_vid_default = args.use_video or (not args.use_tabular and not args.use_video and not args.use_audio)
    use_aud_default = (args.use_audio if has_audio else False) or (not args.use_tabular and not args.use_video and not args.use_audio and has_audio)

    # Class weights
    class_weights_tensor = None
    if args.class_weighted:
        y_series = df_tr[args.label_col].dropna().astype(int)
        counts = y_series.value_counts().reindex(range(num_classes), fill_value=0).values.astype(float)
        with np.errstate(divide="ignore"):
            inv = np.where(counts > 0, 1.0 / counts, 0.0)
        if inv.sum() > 0:
            inv = inv * (len(inv) / max(1.0, inv.sum()))
        class_weights_tensor = torch.tensor(inv, dtype=torch.float32)

    video_kwargs = dict(
        backbone=args.video_backbone,
        backbone_name=args.video_name,
        backbone_trainable=args.video_trainable,
        backbone_unfreeze_last=args.video_unfreeze_last,
        target_size=args.video_target_size,
        lstm_hidden=args.video_lstm_hidden,
        lstm_layers=args.video_lstm_layers,
        bidirectional=args.video_bidirectional,
        dropout=args.video_dropout,
        num_classes=num_classes,
    )

    if args.deterministic:
        model = DeterministicMMVAEAudio(
            tab_in_dim=_to_float_tensor(X_tr_mat).shape[1],
            tab_emb_dim=args.tab_emb,
            audio_emb_dim=args.audio_emb,
            shared_dim=args.shared_dim,
            num_classes=num_classes,
            dropout=args.dropout,
            video_kwargs=video_kwargs,
            classifier_arkoudi=True,
            fuse_dropout=args.fuse_dropout,
            proj_dim=args.proj_dim,
            contrastive_temp=args.contrastive_temp,
            modality_dropout_p=args.modality_dropout,
            fusion_type=args.fusion,
            late_alpha=args.late_alpha,
        )
    else:
        model = VariationalMMVAEAudio(
            tab_in_dim=_to_float_tensor(X_tr_mat).shape[1],
            tab_emb_dim=args.tab_emb,
            audio_emb_dim=args.audio_emb,
            shared_dim=args.shared_dim,
            num_classes=num_classes,
            dropout=args.dropout,
            video_kwargs=video_kwargs,
            classifier_arkoudi=True,
            kl_anneal_steps=args.kl_anneal_steps,
            fuse_dropout=args.fuse_dropout,
            proj_dim=args.proj_dim,
            contrastive_temp=args.contrastive_temp,
            modality_dropout_p=args.modality_dropout,
            fusion_type=args.fusion,
            late_alpha=args.late_alpha,
        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Scheduler
    sched = None
    if args.scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "cosine":
        tmax = args.t_max if args.t_max is not None else args.epochs
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax)
    elif args.scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=args.plateau_patience, factor=args.plateau_factor)

    history = {"loss": [], "acc": []}
    epoch_metrics = []
    global_step = 0
    best_val_acc = -1.0
    best_state = None
    val_hist = []

    overfit_batches = []
    if args.overfit_batches and args.overfit_batches > 0:
        for i, b in enumerate(dl_tr):
            overfit_batches.append(b)
            if len(overfit_batches) >= args.overfit_batches:
                break
    if args.debug_batch:
        if overfit_batches:
            b = overfit_batches[0]
        else:
            b = next(iter(dl_tr))
        print("=== Debug batch ===")
        print("x_tab shape:", b.x_tab.shape, "min/max", b.x_tab.min().item(), b.x_tab.max().item())
        print("x_vid shape:", b.x_vid.shape, "min/max", b.x_vid.min().item(), b.x_vid.max().item())
        if b.x_aud is not None:
            print("x_aud shape:", b.x_aud.shape, "min/max", b.x_aud.min().item(), b.x_aud.max().item())
        else:
            print("x_aud: None")
        print("y shape:", b.y.shape, "labels:", b.y.tolist() if b.y.numel() <= 64 else f"{b.y[:64].tolist()} ...")
        return

    for epoch in range(args.epochs):
        model.train()
        tr_loss, tr_total, tr_correct = 0.0, 0, 0
        tr_align = tr_con = tr_cls = tr_rec_tab = tr_rec_vid = tr_kl = tr_aux_tab = tr_aux_vid = tr_aux_aud = 0.0
        tr_batches = 0
        train_iter = overfit_batches if overfit_batches else dl_tr
        for b in train_iter:
            x_tab = b.x_tab.to(device)
            x_vid = b.x_vid.to(device)
            x_aud = b.x_aud.to(device) if b.x_aud is not None else None
            y = b.y.to(device)

            use_tab = use_tab_default
            use_vid = use_vid_default
            use_aud = use_aud_default
            if epoch < int(args.warmup_epochs):
                if args.warmup_modality == "tabular":
                    use_tab, use_vid, use_aud = True, False, False
                elif args.warmup_modality == "video":
                    use_tab, use_vid, use_aud = False, True, False
                elif args.warmup_modality == "audio":
                    use_tab, use_vid, use_aud = False, False, True
            if not (use_tab or use_vid or use_aud):
                raise SystemExit("Debe estar activa al menos una modalidad")

            optimizer.zero_grad(set_to_none=True)
            ls_now = args.label_smoothing if epoch >= int(args.warmup_epochs) else 0.0
            orig_moddrop = getattr(model, "modality_dropout_p", 0.0)
            if epoch < int(args.warmup_epochs):
                setattr(model, "modality_dropout_p", 0.0)

            if not use_tab:
                x_tab = torch.zeros_like(x_tab)
            if not use_vid:
                x_vid = torch.zeros_like(x_vid)
            if not use_aud and x_aud is not None:
                x_aud = torch.zeros_like(x_aud)

            out = model(x_tab, x_vid, x_aud)
            w_align = 0.0 if (args.warmup_disable_contrastive and epoch < int(args.warmup_epochs)) else args.w_align
            w_con = 0.0 if (args.warmup_disable_contrastive and epoch < int(args.warmup_epochs)) else args.w_contrastive
            w_rec_tab = args.w_rec_tab if use_tab else 0.0
            w_rec_vid = args.w_rec_vid if use_vid else 0.0
            cw = class_weights_tensor.to(device) if class_weights_tensor is not None else None
            w_cls_now = args.w_cls if epoch >= int(args.warmup_epochs) else 0.0

            if isinstance(model, VariationalMMVAEAudio):
                loss, logs = model.loss(
                    out,
                    y=y,
                    w_rec_tab=w_rec_tab,
                    w_rec_vid=w_rec_vid,
                    w_cls=w_cls_now,
                    w_kl=args.w_kl,
                    step=global_step,
                    label_smoothing=ls_now,
                    w_align=w_align,
                    w_contrastive=w_con,
                    w_aux_tab=args.w_aux_tab,
                    w_aux_vid=args.w_aux_vid,
                    w_aux_aud=args.w_aux_aud,
                    class_weights=cw,
                )
            else:
                loss, logs = model.loss(
                    out,
                    y=y,
                    w_rec_tab=w_rec_tab,
                    w_rec_vid=w_rec_vid,
                    w_cls=w_cls_now,
                    label_smoothing=ls_now,
                    w_align=w_align,
                    w_contrastive=w_con,
                    w_aux_tab=args.w_aux_tab,
                    w_aux_vid=args.w_aux_vid,
                    w_aux_aud=args.w_aux_aud,
                    class_weights=cw,
                )

            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            optimizer.step()
            if epoch < int(args.warmup_epochs):
                setattr(model, "modality_dropout_p", orig_moddrop)

            tr_loss += float(loss.item())
            pred = out["logits"].argmax(dim=1)
            tr_correct += int((pred == y).sum().item())
            tr_total += int(y.numel())
            tr_align += float(logs.get("align", 0.0))
            tr_con += float(logs.get("con", 0.0))
            tr_cls += float(logs.get("cls", 0.0))
            tr_rec_tab += float(logs.get("rec_tab", 0.0))
            tr_rec_vid += float(logs.get("rec_vid", 0.0))
            tr_kl += float(logs.get("kl", 0.0))
            tr_aux_tab += float(logs.get("aux_tab", 0.0))
            tr_aux_vid += float(logs.get("aux_vid", 0.0))
            tr_aux_aud += float(logs.get("aux_aud", 0.0))
            tr_batches += 1
            global_step += 1
        tr_acc = tr_correct / max(1, tr_total)
        tr_loss_avg = tr_loss / max(1, tr_batches if overfit_batches else len(dl_tr))
        # nan guard
        tr_loss_avg = float(np.nan_to_num(tr_loss_avg))
        tr_acc = float(np.nan_to_num(tr_acc))
        history["loss"].append(tr_loss_avg)
        history["acc"].append(tr_acc)

        # Validation
        model.eval()
        v_total, v_correct = 0, 0
        v_probs = []
        with torch.no_grad():
            for b in dl_val:
                x_tab = b.x_tab.to(device)
                x_vid = b.x_vid.to(device)
                x_aud = b.x_aud.to(device) if b.x_aud is not None else None
                y = b.y.to(device)
                if not use_tab_default:
                    x_tab = torch.zeros_like(x_tab)
                if not use_vid_default:
                    x_vid = torch.zeros_like(x_vid)
                if not use_aud_default and x_aud is not None:
                    x_aud = torch.zeros_like(x_aud)
                out = model(x_tab, x_vid, x_aud)
                logits = out["logits"]
                v_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                pred = logits.argmax(dim=1)
                v_correct += int((pred == y).sum().item())
                v_total += int(y.numel())
        val_acc = v_correct / max(1, v_total)
        val_hist.append(val_acc)
        if sched is not None:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(val_acc)
            else:
                sched.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        avg_align = float(np.nan_to_num(tr_align / max(1, tr_batches)))
        avg_con = float(np.nan_to_num(tr_con / max(1, tr_batches)))
        avg_cls = float(np.nan_to_num(tr_cls / max(1, tr_batches)))
        avg_rec_tab = float(np.nan_to_num(tr_rec_tab / max(1, tr_batches)))
        avg_rec_vid = float(np.nan_to_num(tr_rec_vid / max(1, tr_batches)))
        avg_kl = float(np.nan_to_num(tr_kl / max(1, tr_batches)))
        avg_aux_tab = float(np.nan_to_num(tr_aux_tab / max(1, tr_batches)))
        avg_aux_vid = float(np.nan_to_num(tr_aux_vid / max(1, tr_batches)))
        avg_aux_aud = float(np.nan_to_num(tr_aux_aud / max(1, tr_batches)))
        cur_lr = optimizer.param_groups[0]["lr"]
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": history["loss"][-1],
                "train_acc": tr_acc,
                "val_acc": float(np.nan_to_num(val_acc)),
                "align_loss": avg_align,
                "contrastive_loss": avg_con,
                "cls_loss": avg_cls,
                "rec_tab_loss": avg_rec_tab,
                "rec_vid_loss": avg_rec_vid,
                "kl_loss": avg_kl,
                "aux_tab_loss": avg_aux_tab,
                "aux_vid_loss": avg_aux_vid,
                "aux_aud_loss": avg_aux_aud,
                "lr": float(np.nan_to_num(cur_lr)),
            }
        )
        print(
            f"Epoch {epoch+1}/{args.epochs} | train_loss={history['loss'][-1]:.4f} | train_acc={tr_acc:.3f} | val_acc={val_acc:.3f} | align={avg_align:.3f} | con={avg_con:.3f} | aux_tab={avg_aux_tab:.3f} | aux_vid={avg_aux_vid:.3f} | aux_aud={avg_aux_aud:.3f} | lr={cur_lr:.2e}"
        )

        if len(val_hist) >= int(args.early_stop_patience):
            window = val_hist[-int(args.early_stop_patience) :]
            if (max(window) - min(window)) <= float(args.early_stop_delta):
                print(f"Early stop: val_acc estable en ±{args.early_stop_delta} durante {args.early_stop_patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Results
    model_name = "MMVAEAudio_Det" if args.deterministic else "MMVAEAudio_Var"
    cfg = {
        "pkl": str(pkl_path),
        "label_col": args.label_col,
        "path_col": args.path_col,
        "audio_col": args.audio_col,
        "features": tab_cols,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "tab_emb": args.tab_emb,
        "shared_dim": args.shared_dim,
        "audio_emb": args.audio_emb,
        "dropout": args.dropout,
        "w_rec_tab": args.w_rec_tab,
        "w_rec_vid": args.w_rec_vid,
        "w_cls": args.w_cls,
        "w_kl": args.w_kl,
        "kl_anneal_steps": args.kl_anneal_steps,
        "fusion": args.fusion,
        "late_alpha": args.late_alpha,
        "w_align": args.w_align,
        "w_contrastive": args.w_contrastive,
        "contrastive_temp": args.contrastive_temp,
        "proj_dim": args.proj_dim,
        "modality_dropout": args.modality_dropout,
        "use_tabular": use_tab_default,
        "use_video": use_vid_default,
        "use_audio": use_aud_default,
        "class_weighted": args.class_weighted,
        "warmup_epochs": args.warmup_epochs,
        "warmup_modality": args.warmup_modality,
        "warmup_disable_contrastive": args.warmup_disable_contrastive,
        "tabular_scaler": args.tabular_scaler,
        "video_norm": args.video_norm,
        "audio_sr": args.audio_sr,
        "audio_duration": args.audio_duration,
        "audio_norm": args.audio_norm,
    }
    run_hash = compute_run_hash(cfg, sys.argv, model=model_name)
    ensure_dir(results_dir)
    torch.save(model.state_dict(), results_dir / artifact_name(model_name, "model", run_hash, "pt"))
    pd.DataFrame(history).to_csv(results_dir / artifact_name(model_name, "history", run_hash, "csv"), index=False)
    if epoch_metrics:
        pd.DataFrame(epoch_metrics).to_csv(results_dir / artifact_name(model_name, "metrics", run_hash, "csv"), index=False)

    # Validation report + confusion matrix + embeddings
    if len(df_val) > 0:
        all_true, all_pred, all_probs = [], [], []
        model.eval()
        with torch.no_grad():
            for b in dl_val:
                x_tab = b.x_tab.to(device)
                x_vid = b.x_vid.to(device)
                x_aud = b.x_aud.to(device) if b.x_aud is not None else None
                y = b.y.to(device)
                if not use_tab_default:
                    x_tab = torch.zeros_like(x_tab)
                if not use_vid_default:
                    x_vid = torch.zeros_like(x_vid)
                if not use_aud_default and x_aud is not None:
                    x_aud = torch.zeros_like(x_aud)
                out = model(x_tab, x_vid, x_aud)
                logits = out["logits"]
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                pred = logits.argmax(dim=1).cpu().numpy().tolist()
                all_true.extend(y.cpu().numpy().tolist())
                all_pred.extend(pred)
                all_probs.append(probs)
        report = classification_report(all_true, all_pred, zero_division=0)
        print("\n=== Validation (Multimodal Audio VAE) ===")
        print(report)
        save_text(report, results_dir / artifact_name(model_name, "eval_report", run_hash, "txt"))
        if all_probs:
            probs = np.concatenate(all_probs, axis=0)
            pd.DataFrame(probs, columns=[f"class_{i}" for i in range(probs.shape[1])]).to_csv(
                results_dir / artifact_name(model_name, "eval_proba", run_hash, "csv"), index=False
            )
        try:
            cm = confusion_matrix(all_true, all_pred)
            pd.DataFrame(cm).to_csv(results_dir / artifact_name(model_name, "confusion_matrix", run_hash, "csv"), index=False)
        except Exception as e:
            save_text(f"confusion_matrix failed: {e}", results_dir / artifact_name(model_name, "confusion_matrix_error", run_hash, "txt"))

    # Save config
    (results_dir / artifact_name(model_name, "config", run_hash, "json")).write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    register_run(results_dir, run_hash, model_name, cmd=" ".join(sys.argv), config=cfg)


if __name__ == "__main__":
    main()
