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
from utils.splits import split_by_participant, format_split_report
from utils.metrics_eval import classification_report_basic, save_metrics

# Ensure package root on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loading.multimodal import MultimodalDataset, collate_multimodal
from src.models.mm_vae import DeterministicMMVAE, VariationalMMVAE
from src.features.prepare import build_preprocessor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.metrics import confusion_matrix
from src.data_cleaning.cleaning import categorias_a_str, convertir_a_categorico
from utils.results_io import (
    ensure_dir,
    save_text,
    save_model_pickle,
    compute_run_hash,
    artifact_name,
    register_run,
)
from utils.features import load_features_file


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
    ap = argparse.ArgumentParser(description="Train Multimodal VAE (tabular + video) end-to-end")
    ap.add_argument("--pkl", type=str, default="data/processed/multimodal_av_join_audio_cached.pkl", help="Ruta al pickle multimodal")
    ap.add_argument("--label-col", type=str, default="action_proc")
    ap.add_argument("--features", nargs="*", default=None, help="Columnas tabulares a usar")
    ap.add_argument(
        "--features-file",
        type=str,
        default="utils/feature_sets/exp1.json",
        help="Archivo con lista de features (json o txt)",
    )
    ap.add_argument("--path-col", type=str, default="frames_route")
    ap.add_argument("--timestamp-col", type=str, default="timestamp")
    ap.add_argument("--window-id-col", type=str, default="window")
    ap.add_argument("--participant-col", type=str, default="participant")
    ap.add_argument("--participant-frac", type=float, default=0.5, help="Fracción de participantes a usar (para acelerar). 0-1.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--test-split", type=float, default=0.0)
    ap.add_argument("--deterministic", action="store_true", help="Usar VAE determinista (por defecto, variacional)")
    ap.add_argument("--tab-emb", type=int, default=128)
    ap.add_argument("--shared-dim", type=int, default=64)
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
    # Early stopping por estabilidad de val_acc
    ap.add_argument("--early-stop-patience", type=int, default=3, help="Corta si en las últimas N epochs la val_acc varía <= delta")
    ap.add_argument("--early-stop-delta", type=float, default=0.02, help="Umbral de variación de val_acc para early stop")
    # Video backbone fine-tuning controls
    ap.add_argument("--video-backbone", type=str, default="vit", choices=["vit", "clip"])
    ap.add_argument("--video-name", type=str, default="vit_b_16")
    ap.add_argument("--video-trainable", action="store_true", default=True)
    ap.add_argument("--freeze-video", action="store_true", help="Congela el backbone de video (ignora video-trainable)")
    ap.add_argument("--video-unfreeze-last", type=int, default=1)
    ap.add_argument("--video-target-size", type=int, default=224)
    ap.add_argument("--video-lstm-hidden", type=int, default=256)
    ap.add_argument("--video-lstm-layers", type=int, default=1)
    ap.add_argument("--video-bidirectional", action="store_true")
    ap.add_argument("--video-dropout", type=float, default=0.0)
    # Fuse dropout
    ap.add_argument("--fuse-dropout", type=float, default=0.1)
    # Fusion mode
    ap.add_argument("--fusion", type=str, default="early", choices=["early", "late"], help="Tipo de fusion: early o late")
    ap.add_argument("--late-alpha", type=float, default=0.5, help="Peso de logits_tab en late fusion (0-1)")
    # Normalizacion
    ap.add_argument("--tabular-scaler", type=str, default="robust", choices=["standard", "robust"], help="Scaler para tabular")
    ap.add_argument("--video-norm", type=str, default="imagenet", choices=["imagenet", "per_channel", "none"], help="Normalizacion para video")
    # Multimodal alignment/contrastive options
    ap.add_argument("--w-align", type=float, default=0.0, help="Peso de pérdida de alineación coseno entre modalidades")
    ap.add_argument("--w-contrastive", type=float, default=0.0, help="Peso de pérdida contrastiva (InfoNCE) entre modalidades")
    ap.add_argument("--contrastive-temp", type=float, default=0.07, help="Temperatura para InfoNCE")
    ap.add_argument("--proj-dim", type=int, default=128, help="Dimensión de proyección para pérdidas de alineación/contrastiva")
    ap.add_argument("--modality-dropout", type=float, default=0.0, help="Probabilidad de apagar una modalidad por muestra en la fusión")
    # Modality toggles (unimodal evaluation)
    ap.add_argument("--use-tabular", action="store_true", help="Usar modalidad tabular en entrenamiento/validación")
    ap.add_argument("--use-video", action="store_true", help="Usar modalidad video en entrenamiento/validación")
    # Overfit mode for debugging
    ap.add_argument("--overfit-batches", type=int, default=0, help="Si >0, sobreajustar a los primeros N minibatches")
    # Class weighting
    ap.add_argument("--class-weighted", action="store_true", default=True, help="CrossEntropy con pesos inversos a la frecuencia de clase")
    # Warmup
    ap.add_argument("--warmup-epochs", type=int, default=0, help="Epochs iniciales con warmup")
    ap.add_argument("--warmup-modality", type=str, default="both", choices=["both", "tabular", "video"], help="Modalidad activa durante warmup")
    ap.add_argument("--warmup-disable-contrastive", action="store_true", help="Desactivar alineación/contrastiva durante warmup")
    args = ap.parse_args()

    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        raise FileNotFoundError(f"No existe el pickle multimodal: {pkl_path}")
    df = pd.read_pickle(pkl_path).reset_index(drop=True)

    # Resolve features (prioriza --features-file, por defecto utils/feature_sets/exp1.json)
    tab_cols = args.features
    if args.features_file:
        loaded = load_features_file(args.features_file)
        if loaded:
            tab_cols = loaded
    if not tab_cols:
        # Heuristic: drop known non-tabular columns
        drop_cols = {args.path_col, args.label_col, args.timestamp_col, args.window_id_col, 'participant', 'session_id'}
        tab_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    # Intersect with available columns to be safe
    tab_cols = [c for c in tab_cols if c in df.columns]

    # Label mapping (usar action_proc por defecto; mapear strings si aparecen)
    default_class_map = {
        'accelerate': 0,
        'brake': 1,
        'decelerate': 2,
        'maintain speed': 3,
        'wait': 4,
    }
    if args.label_col not in df.columns:
        raise KeyError(f"No se encontró la columna de etiqueta '{args.label_col}' en el pickle multimodal")
    if df[args.label_col].dtype == object:
        df[args.label_col] = df[args.label_col].map(default_class_map)
    num_classes = int(pd.Series(df[args.label_col]).nunique())

    # Submuestreo de participantes para acelerar (por defecto 50%)
    if 0 < args.participant_frac < 1.0:
        rng = np.random.RandomState(args.seed)
        parts = pd.Index(df[args.participant_col].dropna().unique())
        k = max(1, int(np.ceil(len(parts) * args.participant_frac)))
        keep_parts = rng.choice(parts, size=k, replace=False)
        df = df[df[args.participant_col].isin(keep_parts)].reset_index(drop=True)
        print(f"Subconjunto de participantes: {len(keep_parts)}/{len(parts)} (frac={args.participant_frac})")

    # Split
    df_tr, df_val, df_te, info = split_by_participant(
        df, participant_col=args.participant_col, val_frac=args.val_split, test_frac=args.test_split, seed=args.seed
    )
    print(format_split_report(info))

    # Preprocesamiento tabular (StandardScaler + OneHot) similar al pipeline baseline
    X_tr_raw = df_tr[tab_cols].copy()
    X_val_raw = df_val[tab_cols].copy()
    X_te_raw = df_te[tab_cols].copy() if len(df_te) else pd.DataFrame(columns=tab_cols)
    # Convertir objetos a categorías para que OneHotEncoder las procese
    X_tr_prep = convertir_a_categorico(categorias_a_str(X_tr_raw))
    X_val_prep = convertir_a_categorico(categorias_a_str(X_val_raw))
    X_te_prep = convertir_a_categorico(categorias_a_str(X_te_raw)) if len(df_te) else X_te_raw
    # Build preprocessor with chosen scaler
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
    X_te_mat = preprocessor.transform(X_te_prep) if len(df_te) else None

    # Persistir preprocessor para reproducibilidad
    results_dir = Path("results")
    ensure_dir(results_dir)
    tmp_cfg = {
        "pkl": str(pkl_path),
        "features": tab_cols,
        "label_col": args.label_col,
    }
    pre_hash = compute_run_hash(tmp_cfg, sys.argv, model="MMVAE_Preproc")
    save_model_pickle(preprocessor, results_dir / artifact_name("MMVAE", "preprocessor", pre_hash, "pkl"))

    # Datasets / loaders
    # Video normalization transform
    def _video_transform(x: torch.Tensor) -> torch.Tensor:
        if args.video_norm == "none":
            return x
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if args.video_norm == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1,3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1,3,1,1)
            if x.size(1) == 3:
                x = (x - mean) / std
            return x if x.size(0) > 1 else x.squeeze(0)
        if args.video_norm == "per_channel":
            eps = 1e-6
            mean = x.mean(dim=(0,2,3), keepdim=True)
            std = x.std(dim=(0,2,3), keepdim=True)
            x = (x - mean) / (std + eps)
            return x if x.size(0) > 1 else x.squeeze(0)
        return x

    ds_tr = MultimodalDataset(
        df_tr,
        tab_columns=tab_cols,
        X_tab_array=_to_float_tensor(X_tr_mat),
        path_col=args.path_col,
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        window_id_col=args.window_id_col,
        prefer_df_label=True,
        class_map=default_class_map,
        video_transform=_video_transform,
    )
    ds_val = MultimodalDataset(
        df_val,
        tab_columns=tab_cols,
        X_tab_array=_to_float_tensor(X_val_mat),
        path_col=args.path_col,
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        window_id_col=args.window_id_col,
        prefer_df_label=True,
        class_map=default_class_map,
        video_transform=_video_transform,
    )
    ds_te = None
    dl_te = None
    if len(df_te):
        ds_te = MultimodalDataset(
            df_te,
            tab_columns=tab_cols,
            X_tab_array=_to_float_tensor(X_te_mat),
            path_col=args.path_col,
            label_col=args.label_col,
            timestamp_col=args.timestamp_col,
            window_id_col=args.window_id_col,
            prefer_df_label=True,
            class_map=default_class_map,
            video_transform=_video_transform,
        )

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_multimodal)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_multimodal)
    if ds_te is not None:
        dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_multimodal)

    # Set defaults for modality usage: if none specified, use both
    use_tab_default = args.use_tabular or (not args.use_tabular and not args.use_video)
    use_vid_default = args.use_video or (not args.use_tabular and not args.use_video)

    # Class weights
    class_weights_tensor = None
    if args.class_weighted:
        y_series = df_tr[args.label_col].dropna().astype(int)
        num_classes = int(pd.Series(df[args.label_col]).nunique())
        counts = y_series.value_counts().reindex(range(num_classes), fill_value=0).values.astype(float)
        with np.errstate(divide='ignore'):
            inv = np.where(counts > 0, 1.0 / counts, 0.0)
        if inv.sum() > 0:
            inv = inv * (len(inv) / max(1.0, inv.sum()))
        class_weights_tensor = torch.tensor(inv, dtype=torch.float32)

    # Model
    tab_in_dim = ds_tr.X_tab_array.shape[1]
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
    if args.deterministic:
        model = DeterministicMMVAE(
            tab_in_dim=tab_in_dim,
            tab_emb_dim=args.tab_emb,
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
        model = VariationalMMVAE(
            tab_in_dim=tab_in_dim,
            tab_emb_dim=args.tab_emb,
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
    global_step = 0
    best_val_acc = -1.0
    best_state = None
    val_hist = []
    # Overfit batches setup
    overfit_batches = []
    if args.overfit_batches and args.overfit_batches > 0:
        for i, b in enumerate(dl_tr):
            overfit_batches.append(b)
            if len(overfit_batches) >= args.overfit_batches:
                break
    
    epoch_metrics = []
    for epoch in range(args.epochs):
        model.train()
        tr_loss, tr_total, tr_correct = 0.0, 0, 0
        tr_align, tr_con, tr_cls, tr_rec_tab, tr_rec_vid, tr_kl, tr_batches = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        train_iter = overfit_batches if overfit_batches else dl_tr
        for b in train_iter:
            x_tab = b.x_tab.to(device)
            x_vid = b.x_vid.to(device)
            y = b.y.to(device)
            if x_vid.dim() == 5:
                x_vid = x_vid[:, :1]  # usa solo el primer frame para acelerar

            # Warmup and modality toggles
            use_tab = use_tab_default
            use_vid = use_vid_default
            if epoch < int(args.warmup_epochs):
                if args.warmup_modality == "tabular":
                    use_tab, use_vid = True, False
                elif args.warmup_modality == "video":
                    use_tab, use_vid = False, True
            if not use_tab and not use_vid:
                raise SystemExit("Debe estar activa al menos una modalidad (tabular o video)")

            optimizer.zero_grad(set_to_none=True)
            # Apply modality masks by zeroing inputs of disabled modality
            if not use_tab:
                x_tab = torch.zeros_like(x_tab)
            if not use_vid:
                x_vid = torch.zeros_like(x_vid)
            out = model(x_tab, x_vid)
            # Label smoothing active only after warmup
            ls_now = args.label_smoothing if epoch >= int(args.warmup_epochs) else 0.0
            # Temporarily disable modality dropout during warmup
            orig_moddrop = getattr(model, 'modality_dropout_p', 0.0)
            if epoch < int(args.warmup_epochs):
                setattr(model, 'modality_dropout_p', 0.0)

            if isinstance(model, VariationalMMVAE):
                w_align = 0.0 if (args.warmup_disable_contrastive and epoch < int(args.warmup_epochs)) else args.w_align
                w_con = 0.0 if (args.warmup_disable_contrastive and epoch < int(args.warmup_epochs)) else args.w_contrastive
                # Disable recon losses for masked modalities
                w_rec_tab = args.w_rec_tab if use_tab else 0.0
                w_rec_vid = args.w_rec_vid if use_vid else 0.0
                cw = class_weights_tensor.to(device) if class_weights_tensor is not None else None
                loss, logs = model.loss(out, y=y, w_rec_tab=w_rec_tab, w_rec_vid=w_rec_vid, w_cls=args.w_cls, w_kl=args.w_kl, step=global_step, label_smoothing=ls_now, w_align=w_align, w_contrastive=w_con, class_weights=cw)
            else:
                w_align = 0.0 if (args.warmup_disable_contrastive and epoch < int(args.warmup_epochs)) else args.w_align
                w_con = 0.0 if (args.warmup_disable_contrastive and epoch < int(args.warmup_epochs)) else args.w_contrastive
                w_rec_tab = args.w_rec_tab if use_tab else 0.0
                w_rec_vid = args.w_rec_vid if use_vid else 0.0
                cw = class_weights_tensor.to(device) if class_weights_tensor is not None else None
                loss, logs = model.loss(out, y=y, w_rec_tab=w_rec_tab, w_rec_vid=w_rec_vid, w_cls=args.w_cls, label_smoothing=ls_now, w_align=w_align, w_contrastive=w_con, class_weights=cw)
            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            optimizer.step()
            # Restore modality dropout after step
            if epoch < int(args.warmup_epochs):
                setattr(model, 'modality_dropout_p', orig_moddrop)
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
                y = b.y.to(device)
                if x_vid.dim() == 5:
                    x_vid = x_vid[:, :1]
                if not use_tab_default:
                    x_tab = torch.zeros_like(x_tab)
                if not use_vid_default:
                    x_vid = torch.zeros_like(x_vid)
                out = model(x_tab, x_vid)
                logits = out["logits"]
                v_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                pred = logits.argmax(dim=1)
                v_correct += int((pred == y).sum().item())
                v_total += int(y.numel())
        val_acc = v_correct / max(1, v_total)
        val_hist.append(val_acc)

        # Scheduler step
        if sched is not None:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(val_acc)
            else:
                sched.step()
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        # Aggregate aux losses and lr
        avg_align = (tr_align / max(1, tr_batches))
        avg_con = (tr_con / max(1, tr_batches))
        avg_cls = (tr_cls / max(1, tr_batches))
        avg_rec_tab = (tr_rec_tab / max(1, tr_batches))
        avg_rec_vid = (tr_rec_vid / max(1, tr_batches))
        avg_kl = (tr_kl / max(1, tr_batches))
        cur_lr = optimizer.param_groups[0]["lr"]
        epoch_metrics.append({
            "epoch": epoch + 1,
            "train_loss": history['loss'][-1],
            "train_acc": tr_acc,
            "val_acc": val_acc,
            "align_loss": avg_align,
            "contrastive_loss": avg_con,
            "cls_loss": avg_cls,
            "rec_tab_loss": avg_rec_tab,
            "rec_vid_loss": avg_rec_vid,
            "kl_loss": avg_kl,
            "lr": cur_lr,
        })
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={history['loss'][-1]:.4f} | train_acc={tr_acc:.3f} | val_acc={val_acc:.3f} | align={avg_align:.3f} | con={avg_con:.3f} | lr={cur_lr:.2e}")

        # Early stopping: si en las últimas N epochs la variación <= delta
        if len(val_hist) >= int(args.early_stop_patience):
            window = val_hist[-int(args.early_stop_patience):]
            if (max(window) - min(window)) <= float(args.early_stop_delta):
                print(f"Early stop: val_acc estable en ±{args.early_stop_delta} durante {args.early_stop_patience} epochs.")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Results
    results_dir = Path("results")
    ensure_dir(results_dir)
    model_name = "MMVAE_Det" if args.deterministic else "MMVAE_Var"
    cfg = {
        "pkl": str(pkl_path),
        "label_col": args.label_col,
        "path_col": args.path_col,
        "features": tab_cols,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "tab_emb": args.tab_emb,
        "shared_dim": args.shared_dim,
        "dropout": args.dropout,
        "w_rec_tab": args.w_rec_tab,
        "w_rec_vid": args.w_rec_vid,
        "w_cls": args.w_cls,
        "w_kl": args.w_kl,
        "kl_anneal_steps": args.kl_anneal_steps,
        "participant_col": args.participant_col,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "freeze_video": args.freeze_video,
        "argv": sys.argv,
    }
    run_hash = compute_run_hash(cfg, sys.argv, model=model_name)
    torch.save(model.state_dict(), results_dir / artifact_name(model_name, "model", run_hash, "pt"))
    pd.DataFrame(history).to_csv(results_dir / artifact_name(model_name, "history", run_hash, "csv"), index=False)
    if epoch_metrics:
        pd.DataFrame(epoch_metrics).to_csv(results_dir / artifact_name(model_name, "metrics", run_hash, "csv"), index=False)
    # Save preprocessor tied to this run as well
    save_model_pickle(preprocessor, results_dir / artifact_name(model_name, "preprocessor", run_hash, "pkl"))

    # Extraer y guardar embeddings finales (para análisis econométrico)
    X_all_raw = df[tab_cols].copy()
    X_all_prep = convertir_a_categorico(categorias_a_str(X_all_raw))
    X_all_mat = preprocessor.transform(X_all_prep)
    ds_all = MultimodalDataset(
        df,
        tab_columns=tab_cols,
        X_tab_array=_to_float_tensor(X_all_mat),
        path_col=args.path_col,
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        window_id_col=args.window_id_col,
        prefer_df_label=True,
        class_map=default_class_map,
    )
    dl_all = DataLoader(ds_all, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_multimodal)
    model.eval()
    zs, mus, logvars, ys_all, ts_all, wids_all, parts_all = [], [], [], [], [], [], []
    with torch.no_grad():
        for b in dl_all:
            x_tab = b.x_tab.to(device)
            x_vid = b.x_vid.to(device)
            if x_vid.dim() == 5:
                x_vid = x_vid[:, :1]
            out = model(x_tab, x_vid)
            z = out["z"].detach().cpu().numpy()
            zs.append(z)
            if "mu" in out and "logvar" in out:
                mus.append(out["mu"].detach().cpu().numpy())
                logvars.append(out["logvar"].detach().cpu().numpy())
            ys_all.extend(b.y.numpy().tolist())
            ts_all.extend(b.timestamp)
            wids_all.extend(b.window_id)
            parts_all.extend(b.participant)
    z_mat = np.concatenate(zs, axis=0)
    z_cols = [f"z_{i}" for i in range(z_mat.shape[1])]
    base_df = pd.DataFrame(z_mat, columns=z_cols)
    meta_df = pd.DataFrame({
        "label": ys_all,
        "timestamp": ts_all,
        "window_id": wids_all,
        "participant": parts_all,
    })
    concat_list = [base_df, meta_df]
    if mus:
        mu_mat = np.concatenate(mus, axis=0)
        lv_mat = np.concatenate(logvars, axis=0) if logvars else np.zeros_like(mu_mat)
        std_mat = np.exp(0.5 * lv_mat)
        mu_df = pd.DataFrame(mu_mat, columns=[f"mu_{i}" for i in range(mu_mat.shape[1])])
        std_df = pd.DataFrame(std_mat, columns=[f"std_{i}" for i in range(std_mat.shape[1])])
        concat_list.insert(1, mu_df)
        concat_list.insert(2, std_df)
    emb_df = pd.concat(concat_list, axis=1)
    emb_df.to_csv(results_dir / artifact_name(model_name, "embeddings", run_hash, "csv"), index=False)

    # Save run config and metrics
    val_metrics = classification_report_basic(np.array(all_true := []), np.array(all_pred := []), None) if False else {}
    # Re-evaluate val/test for metrics with log_probs
    def eval_loader(loader):
        ys, preds, logps = [], [], []
        model.eval()
        with torch.no_grad():
            for b in loader:
                x_tab = b.x_tab.to(device)
                x_vid = b.x_vid.to(device)
                y = b.y.to(device)
                if not use_tab_default:
                    x_tab = torch.zeros_like(x_tab)
                if not use_vid_default:
                    x_vid = torch.zeros_like(x_vid)
                out = model(x_tab, x_vid)
                logits = out["logits"]
                ys.append(y.cpu())
                lp = torch.log_softmax(logits, dim=1).cpu()
                logps.append(lp)
                preds.append(lp.argmax(dim=1))
        if not ys:
            return {}
        y_true = torch.cat(ys).numpy()
        y_pred = torch.cat(preds).numpy()
        logp_np = torch.cat(logps).numpy()
        return classification_report_basic(y_true, y_pred, log_probs=logp_np)

    metrics_val = eval_loader(dl_val)
    metrics_test = eval_loader(dl_te) if dl_te is not None else {}
    all_metrics = {f"val_{k}": v for k, v in metrics_val.items()}
    all_metrics.update({f"test_{k}": v for k, v in metrics_test.items()})
    save_metrics(all_metrics, results_dir, model_name=model_name, config=cfg)
    split_path = results_dir / model_name / "split_info.txt"
    split_path.write_text(format_split_report(info), encoding="utf-8")
    (results_dir / artifact_name(model_name, "config", run_hash, "json")).write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    register_run(results_dir, run_hash, model_name, cmd=" ".join(sys.argv), config=cfg)
    print(f"Resultados guardados en: {results_dir / model_name}")


if __name__ == "__main__":
    main()
