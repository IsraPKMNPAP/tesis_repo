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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler

# Ensure package root on path (dataset_bicicletas/)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loading.multimodal_audio import MultimodalAudioDataset, collate_multimodal_audio
from src.models.mm_vae_audio_interpretable import (
    InterpretableMMVAEAudioDeterministic,
    InterpretableMMVAEAudioVariational,
)
from utils.splits import split_by_participant, format_split_report
from utils.metrics_eval import classification_report_basic, save_metrics
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


def main():
    ap = argparse.ArgumentParser(
        description="Entrena MM-VAE multimodal (tab + video + audio) con embedding interpretable (Arkoudi et al.)"
    )
    # Datos
    ap.add_argument(
        "--pkl",
        type=str,
        default="data/processed/multimodal_av_join_audio_cached.pkl",
        help="Ruta al pickle multimodal (por defecto, el que ya contiene paths cacheados de audio)",
    )
    ap.add_argument("--label-col", type=str, default="action_proc")
    ap.add_argument("--features", nargs="*", default=None, help="Columnas tabulares a usar")
    ap.add_argument("--features-file", type=str, default="utils/feature_sets/exp1.json", help="Archivo con lista de features")
    ap.add_argument("--path-col", type=str, default="frames_route", help="Columna con ruta a frames/tensor de video")
    ap.add_argument("--video-root", type=str, default=None, help="Raíz para prefijar paths de video si son relativos")
    ap.add_argument("--audio-col", type=str, default=None, help="No usada en cache; solo si hubiera wav directo")
    ap.add_argument("--timestamp-col", type=str, default="timestamp")
    ap.add_argument("--window-id-col", type=str, default="window")
    ap.add_argument("--participant-col", type=str, default="participant")
    ap.add_argument("--participant-frac", type=float, default=1.0, help="Fracción de participantes a usar (1.0 = todos)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audio-start-col", type=str, default="audio_segment_start")
    ap.add_argument("--audio-cached-col", type=str, default="audio_cached_path", help="Columna con segmento .pt precalculado")
    ap.add_argument("--audio-root", type=str, default="data/processed/audio_segments_cached", help="Raíz donde viven los segmentos cacheados")
    ap.add_argument("--audio-template", type=str, default="raw_audio_{participant}.wav")
    ap.add_argument("--audio-fallback-template", type=str, default=None)
    # Opt/entrenamiento
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--test-split", type=float, default=0.0)
    ap.add_argument("--deterministic", action="store_true", help="Usar VAE determinista (por defecto, variacional)")
    ap.add_argument("--tab-emb", type=int, default=128)
    ap.add_argument("--shared-dim", type=int, default=None, help="Dim del embedding interpretable; default=num_classes")
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
    ap.add_argument("--scheduler", type=str, default=None, choices=[None, "step", "cosine", "plateau"], help="Scheduler de LR")
    ap.add_argument("--step-size", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--t-max", type=int, default=None)
    ap.add_argument("--plateau-patience", type=int, default=3)
    ap.add_argument("--plateau-factor", type=float, default=0.5)
    ap.add_argument("--early-stop-patience", type=int, default=3)
    ap.add_argument("--early-stop-delta", type=float, default=0.02)
    # Video backbone fine-tuning
    ap.add_argument("--video-backbone", type=str, default="vit", choices=["vit", "clip"])
    ap.add_argument("--video-name", type=str, default="vit_b_16")
    ap.add_argument("--video-trainable", action="store_true", default=True)
    ap.add_argument("--freeze-video", action="store_true", help="Congela el backbone de video")
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
    ap.add_argument("--audio-duration", type=float, default=2.0)
    ap.add_argument("--audio-norm", type=str, default="per_channel", choices=["per_channel", "none"])
    ap.add_argument("--audio-encoder", type=str, default="simple", choices=["simple", "cnn", "tcn", "wav2vec"])
    ap.add_argument("--audio-n-mels", type=int, default=64)
    ap.add_argument("--audio-cnn-channels", nargs="+", type=int, default=[32, 64, 128])
    ap.add_argument("--audio-tcn-channels", nargs="+", type=int, default=[64, 128, 256])
    ap.add_argument("--audio-tcn-kernel", type=int, default=3)
    ap.add_argument("--audio-encoder-dropout", type=float, default=0.2)
    ap.add_argument("--audio-wav2vec-bundle", type=str, default="WAV2VEC2_BASE")
    ap.add_argument("--audio-wav2vec-trainable", action="store_true")
    ap.add_argument("--freeze-audio", action="store_true", help="Congela el encoder de audio")
    # Interpretabilidad
    ap.add_argument("--w-l1-z", type=float, default=0.05, help="Peso L1 sobre z interpretable")
    ap.add_argument("--w-ortho-proto", type=float, default=0.05, help="Peso de ortogonalidad sobre embeddings de clase")
    ap.add_argument("--w-margin", type=float, default=0.05, help="Peso para margin/one-hot en z interpretable")
    ap.add_argument("--margin-type", type=str, default="mse", choices=["mse", "hinge"])
    args = ap.parse_args()

    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        raise FileNotFoundError(f"No existe el pickle multimodal: {pkl_path}")
    df = pd.read_pickle(pkl_path).reset_index(drop=True)

    # Features
    tab_cols = args.features
    if args.features_file:
        loaded = load_features_file(args.features_file)
        if loaded:
            tab_cols = loaded
    if not tab_cols:
        drop_cols = {
            args.path_col,
            args.audio_col,
            args.audio_cached_col,
            args.label_col,
            args.timestamp_col,
            args.window_id_col,
            args.participant_col,
            "session_id",
        }
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
    interpretable_dim = args.shared_dim if args.shared_dim is not None else num_classes

    # Prefijar rutas de video si aplica
    if args.video_root:
        root = Path(args.video_root)
        df[args.path_col] = df[args.path_col].astype(str).apply(lambda p: str(root / p) if not Path(p).is_absolute() else p)
    # Prefijar rutas de audio cacheado si aplica
    if args.audio_cached_col and args.audio_cached_col in df.columns and args.audio_root:
        aroot = Path(args.audio_root)
        df[args.audio_cached_col] = df[args.audio_cached_col].astype(str).apply(
            lambda p: str(aroot / p) if p not in ("", "nan", "None") and not Path(p).is_absolute() else p
        )

    # Submuestreo de participantes si se solicita
    if 0 < args.participant_frac < 1.0:
        rng = np.random.RandomState(args.seed)
        parts = pd.Index(df[args.participant_col].dropna().unique())
        k = max(1, int(np.ceil(len(parts) * args.participant_frac)))
        keep_parts = rng.choice(parts, size=k, replace=False)
        df = df[df[args.participant_col].isin(keep_parts)].reset_index(drop=True)
        print(f"Subconjunto de participantes: {len(keep_parts)}/{len(parts)} (frac={args.participant_frac})")

    # Split por participante
    df_tr, df_val, df_te, info = split_by_participant(
        df, participant_col=args.participant_col, val_frac=args.val_split, test_frac=args.test_split, seed=args.seed
    )
    print(format_split_report(info))

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
    pre_hash = compute_run_hash(tmp_cfg, sys.argv, model="MMVAEAudio_Interp_Preproc")
    save_model_pickle(preprocessor, results_dir / artifact_name("MMVAEAudio_Interp", "preprocessor", pre_hash, "pkl"))

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

    # Audio availability
    has_audio = (bool(args.audio_root) or args.audio_cached_col) and args.audio_duration > 0

    # Datasets / loaders
    ds_tr = MultimodalAudioDataset(
        df_tr,
        tab_columns=tab_cols,
        X_tab_array=_to_float_tensor(X_tr_mat),
        path_col=args.path_col,
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        window_id_col=args.window_id_col,
        participant_col=args.participant_col,
        audio_start_col=args.audio_start_col,
        audio_cached_col=args.audio_cached_col,
        audio_root=args.audio_root if has_audio else None,
        audio_template=args.audio_template,
        audio_fallback_template=args.audio_fallback_template,
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
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        window_id_col=args.window_id_col,
        participant_col=args.participant_col,
        audio_start_col=args.audio_start_col,
        audio_cached_col=args.audio_cached_col,
        audio_root=args.audio_root if has_audio else None,
        audio_template=args.audio_template,
        audio_fallback_template=args.audio_fallback_template,
        prefer_df_label=True,
        class_map=default_class_map,
        video_transform=_video_transform,
        audio_sr=args.audio_sr,
        audio_duration=args.audio_duration,
        audio_norm=args.audio_norm,
    )

    ds_te = None
    dl_te = None
    if len(df_te):
        X_te_prep = convertir_a_categorico(categorias_a_str(df_te[tab_cols])) if len(df_te) else df_te[tab_cols]
        X_te_mat = preprocessor.transform(X_te_prep) if len(df_te) else None
        ds_te = MultimodalAudioDataset(
            df_te,
            tab_columns=tab_cols,
            X_tab_array=_to_float_tensor(X_te_mat) if X_te_mat is not None else None,
            path_col=args.path_col,
            label_col=args.label_col,
            timestamp_col=args.timestamp_col,
            window_id_col=args.window_id_col,
            participant_col=args.participant_col,
            audio_start_col=args.audio_start_col,
            audio_cached_col=args.audio_cached_col,
            audio_root=args.audio_root if has_audio else None,
            audio_template=args.audio_template,
            audio_fallback_template=args.audio_fallback_template,
            prefer_df_label=True,
            class_map=default_class_map,
            video_transform=_video_transform,
            audio_sr=args.audio_sr,
            audio_duration=args.audio_duration,
            audio_norm=args.audio_norm,
        )

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_multimodal_audio)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_multimodal_audio)
    if ds_te:
        dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_multimodal_audio)

    # Modality defaults
    use_tab_default = args.use_tabular or (not args.use_tabular and not args.use_video and not args.use_audio)
    use_vid_default = args.use_video or (not args.use_tabular and not args.use_video and not args.use_audio)
    use_aud_default = (args.use_audio or (not args.use_tabular and not args.use_video and not args.use_audio)) and has_audio

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
        backbone_trainable=bool(args.video_trainable and not args.freeze_video),
        backbone_unfreeze_last=args.video_unfreeze_last,
        target_size=args.video_target_size,
        lstm_hidden=args.video_lstm_hidden,
        lstm_layers=args.video_lstm_layers,
        bidirectional=args.video_bidirectional,
        dropout=args.video_dropout,
        num_classes=num_classes,
    )

    audio_kwargs = dict(
        sample_rate=args.audio_sr,
        n_mels=args.audio_n_mels,
        cnn_channels=args.audio_cnn_channels,
        tcn_channels=args.audio_tcn_channels,
        kernel_size=args.audio_tcn_kernel,
        dropout=args.audio_encoder_dropout,
        bundle_name=args.audio_wav2vec_bundle,
        trainable=bool(args.audio_wav2vec_trainable and not args.freeze_audio),
    )

    model_kwargs = dict(
        tab_in_dim=_to_float_tensor(X_tr_mat).shape[1],
        tab_emb_dim=args.tab_emb,
        audio_emb_dim=args.audio_emb,
        shared_dim=interpretable_dim,
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
        audio_encoder_type=args.audio_encoder,
        audio_kwargs=audio_kwargs,
    )

    if args.deterministic:
        model = InterpretableMMVAEAudioDeterministic(**model_kwargs)
        model_name = "MMVAEAudio_Interp_Det"
    else:
        model = InterpretableMMVAEAudioVariational(**model_kwargs, kl_anneal_steps=args.kl_anneal_steps)
        model_name = "MMVAEAudio_Interp_Var"

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

    for epoch in range(args.epochs):
        model.train()
        tr_loss, tr_total, tr_correct = 0.0, 0, 0
        tr_align = tr_con = tr_cls = tr_rec_tab = tr_rec_vid = tr_kl = tr_aux_tab = tr_aux_vid = tr_aux_aud = tr_l1 = tr_ortho = tr_margin = 0.0
        tr_batches = 0
        train_iter = overfit_batches if overfit_batches else dl_tr
        for b in train_iter:
            x_tab = b.x_tab.to(device)
            x_vid = b.x_vid.to(device)
            x_aud = b.x_aud.to(device) if b.x_aud is not None else None
            y = b.y.to(device)
            if x_vid.dim() == 5:
                x_vid = x_vid[:, :3]
            if x_aud is not None:
                max_len = int(args.audio_sr * args.audio_duration)
                x_aud = x_aud[..., :max_len]

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

            loss_kwargs = dict(
                out=out,
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
                w_l1_z=args.w_l1_z,
                w_proto_ortho=args.w_ortho_proto,
                w_margin=args.w_margin,
                margin_type=args.margin_type,
            )
            if isinstance(model, InterpretableMMVAEAudioVariational):
                loss, logs = model.loss(**loss_kwargs, w_kl=args.w_kl, step=global_step)
            else:
                loss, logs = model.loss(**loss_kwargs)

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
            tr_l1 += float(logs.get("l1_z", 0.0))
            tr_ortho += float(logs.get("proto_ortho", 0.0))
            tr_margin += float(logs.get("margin", 0.0))
            tr_batches += 1
            global_step += 1
        tr_acc = tr_correct / max(1, tr_total)
        history["loss"].append(tr_loss / max(1, len(dl_tr)))
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
                if x_vid.dim() == 5:
                    x_vid = x_vid[:, :3]
                if x_aud is not None:
                    max_len = int(args.audio_sr * args.audio_duration)
                    x_aud = x_aud[..., :max_len]
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

        avg_align = (tr_align / max(1, tr_batches))
        avg_con = (tr_con / max(1, tr_batches))
        avg_cls = (tr_cls / max(1, tr_batches))
        avg_rec_tab = (tr_rec_tab / max(1, tr_batches))
        avg_rec_vid = (tr_rec_vid / max(1, tr_batches))
        avg_kl = (tr_kl / max(1, tr_batches))
        avg_aux_tab = (tr_aux_tab / max(1, tr_batches))
        avg_aux_vid = (tr_aux_vid / max(1, tr_batches))
        avg_aux_aud = (tr_aux_aud / max(1, tr_batches))
        avg_l1 = (tr_l1 / max(1, tr_batches))
        avg_ortho = (tr_ortho / max(1, tr_batches))
        avg_margin = (tr_margin / max(1, tr_batches))
        cur_lr = optimizer.param_groups[0]["lr"]
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": history["loss"][-1],
                "train_acc": tr_acc,
                "val_acc": val_acc,
                "align_loss": avg_align,
                "contrastive_loss": avg_con,
                "cls_loss": avg_cls,
                "rec_tab_loss": avg_rec_tab,
                "rec_vid_loss": avg_rec_vid,
                "kl_loss": avg_kl,
                "aux_tab_loss": avg_aux_tab,
                "aux_vid_loss": avg_aux_vid,
                "aux_aud_loss": avg_aux_aud,
                "l1_z": avg_l1,
                "proto_ortho": avg_ortho,
                "margin_loss": avg_margin,
                "lr": cur_lr,
            }
        )
        print(
            f"Epoch {epoch+1}/{args.epochs} | train_loss={history['loss'][-1]:.4f} | train_acc={tr_acc:.3f} | val_acc={val_acc:.3f} | align={avg_align:.3f} | con={avg_con:.3f} | l1={avg_l1:.3f} | ortho={avg_ortho:.3f} | margin={avg_margin:.3f} | lr={cur_lr:.2e}"
        )

        if len(val_hist) >= int(args.early_stop_patience):
            window = val_hist[-int(args.early_stop_patience) :]
            if (max(window) - min(window)) <= float(args.early_stop_delta):
                print(f"Early stop: val_acc estable en +/-{args.early_stop_delta} durante {args.early_stop_patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Results
    cfg = {
        "pkl": str(pkl_path),
        "label_col": args.label_col,
        "path_col": args.path_col,
        "audio_col": args.audio_col,
        "audio_cached_col": args.audio_cached_col,
        "audio_root": args.audio_root if has_audio else None,
        "audio_start_col": args.audio_start_col,
        "participant_col": args.participant_col,
        "participant_frac": args.participant_frac,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "seed": args.seed,
        "features": tab_cols,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "tab_emb": args.tab_emb,
        "shared_dim": interpretable_dim,
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
        "audio_encoder": args.audio_encoder,
        "audio_n_mels": args.audio_n_mels,
        "audio_cnn_channels": args.audio_cnn_channels,
        "audio_tcn_channels": args.audio_tcn_channels,
        "audio_tcn_kernel": args.audio_tcn_kernel,
        "audio_encoder_dropout": args.audio_encoder_dropout,
        "audio_wav2vec_bundle": args.audio_wav2vec_bundle,
        "audio_wav2vec_trainable": args.audio_wav2vec_trainable,
        "w_l1_z": args.w_l1_z,
        "w_ortho_proto": args.w_ortho_proto,
        "w_margin": args.w_margin,
        "margin_type": args.margin_type,
    }
    run_hash = compute_run_hash(cfg, sys.argv, model=model_name)
    ensure_dir(results_dir)
    torch.save(model.state_dict(), results_dir / artifact_name(model_name, "model", run_hash, "pt"))
    pd.DataFrame(history).to_csv(results_dir / artifact_name(model_name, "history", run_hash, "csv"), index=False)
    if epoch_metrics:
        pd.DataFrame(epoch_metrics).to_csv(results_dir / artifact_name(model_name, "metrics", run_hash, "csv"), index=False)

    def eval_loader(loader):
        if loader is None:
            return {}
        ys, preds, logps = [], [], []
        model.eval()
        with torch.no_grad():
            for b in loader:
                x_tab = b.x_tab.to(device)
                x_vid = b.x_vid.to(device)
                x_aud = b.x_aud.to(device) if b.x_aud is not None else None
                y = b.y.to(device)
                if x_vid.dim() == 5:
                    x_vid = x_vid[:, :3]
                if x_aud is not None:
                    max_len = int(args.audio_sr * args.audio_duration)
                    x_aud = x_aud[..., :max_len]
                if not use_tab_default:
                    x_tab = torch.zeros_like(x_tab)
                if not use_vid_default:
                    x_vid = torch.zeros_like(x_vid)
                if not use_aud_default and x_aud is not None:
                    x_aud = torch.zeros_like(x_aud)
                out = model(x_tab, x_vid, x_aud)
                logits = out["logits"]
                lp = torch.log_softmax(logits, dim=1).cpu()
                ys.append(y.cpu())
                preds.append(lp.argmax(dim=1))
                logps.append(lp)
        if not ys:
            return {}
        y_true = torch.cat(ys).numpy()
        y_pred = torch.cat(preds).numpy()
        logp_np = torch.cat(logps).numpy()
        return classification_report_basic(y_true, y_pred, log_probs=logp_np)

    metrics_val = eval_loader(dl_val)
    metrics_test = eval_loader(dl_te)
    all_metrics = {f"val_{k}": v for k, v in metrics_val.items()}
    all_metrics.update({f"test_{k}": v for k, v in metrics_test.items()})
    save_metrics(all_metrics, results_dir, model_name=model_name, config=cfg, run_hash=run_hash)
    split_path = results_dir / model_name / "split_info.txt"
    split_path.write_text(format_split_report(info), encoding="utf-8")
    (results_dir / artifact_name(model_name, "config", run_hash, "json")).write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    register_run(results_dir, run_hash, model_name, cmd=" ".join(sys.argv), config=cfg)


if __name__ == "__main__":
    main()
