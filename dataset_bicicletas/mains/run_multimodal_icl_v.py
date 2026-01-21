from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loading.multimodal_icl_v import MultimodalICLVDataset, collate_multimodal_icl_v
from src.data_loading.multimodal_audio import MultimodalAudioDataset
from src.models.icl_v import MultimodalICLVDeterministic
from utils.features import load_features_file
from utils.splits import split_by_participant, format_split_report
from utils.metrics_eval import (
    classification_report_basic,
    pseudo_r2_mcfadden,
    save_metrics,
)
from utils.results_io import (
    ensure_dir,
    save_model_pickle,
    compute_run_hash,
    artifact_name,
    register_run,
)


def to_float_array(mat) -> np.ndarray:
    try:
        arr = mat.toarray()
    except Exception:
        arr = np.asarray(mat)
    return arr.astype(np.float32)


def split_cols_by_type(df: pd.DataFrame, cols: Sequence[str], cat_unique_threshold: int):
    numeric = []
    categorical = []
    dropped = []
    for c in cols:
        if c not in df.columns:
            continue
        try:
            nunique = df[c].nunique(dropna=True)
        except Exception:
            nunique = None
        is_num = pd.api.types.is_numeric_dtype(df[c])
        if not is_num and nunique is not None and nunique > cat_unique_threshold:
            dropped.append(c)
            continue
        if nunique is not None and nunique <= cat_unique_threshold:
            categorical.append(c)
        else:
            if is_num:
                numeric.append(c)
            else:
                categorical.append(c)
    return numeric, categorical, dropped


def prepare_preprocessor(df: pd.DataFrame, cols: Sequence[str], cat_unique_threshold: int):
    cols = [c for c in cols if c in df.columns]
    numeric, categorical, dropped = split_cols_by_type(df, cols, cat_unique_threshold)
    if dropped:
        print(f"[prep] dropped high-card categoricals: {dropped}")
    df_num = df[numeric].apply(pd.to_numeric, errors="coerce") if numeric else pd.DataFrame(index=df.index)
    if numeric:
        df_num = df_num.fillna(df_num.median(numeric_only=True))
    df_cat = df[categorical].copy() if categorical else pd.DataFrame(index=df.index)
    if categorical:
        for c in categorical:
            mode = df_cat[c].mode(dropna=True)
            fill_val = mode.iloc[0] if len(mode) else "missing"
            df_cat[c] = df_cat[c].fillna(fill_val).infer_objects(copy=False).astype(str)
    df_prep = pd.concat([df_num, df_cat], axis=1)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical),
        ]
    )
    mat = preprocessor.fit_transform(df_prep)
    kept_cols = numeric + categorical
    return mat, preprocessor, kept_cols


def encode_indicator_blocks(df_tr: pd.DataFrame, df_val: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    tr_blocks = []
    val_blocks = []
    for col in cols:
        if pd.api.types.is_numeric_dtype(df_tr[col]):
            tr_col = df_tr[col].fillna(df_tr[col].median())
            val_col = df_val[col].fillna(df_tr[col].median())
        else:
            tr_str = df_tr[col].astype(str)
            uniq = tr_str.unique().tolist()
            mapping = {v: i for i, v in enumerate(uniq)}
            tr_col = tr_str.map(mapping).fillna(-1)
            val_col = df_val[col].astype(str).map(mapping).fillna(-1)
        tr_blocks.append(tr_col.to_numpy(dtype=np.float32))
        val_blocks.append(val_col.to_numpy(dtype=np.float32))
    if not tr_blocks:
        return np.zeros((len(df_tr), 0), dtype=np.float32), np.zeros((len(df_val), 0), dtype=np.float32)
    tr_mat = np.stack(tr_blocks, axis=1).astype(np.float32)
    val_mat = np.stack(val_blocks, axis=1).astype(np.float32)
    return tr_mat, val_mat


def resolve_cols(df: pd.DataFrame, explicit: Sequence[str] | None, file_path: str | None, drop_cols: set) -> List[str]:
    if explicit:
        cols = list(explicit)
    elif file_path:
        cols = load_features_file(file_path)
    else:
        cols = []
    if not cols:
        cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if c in df.columns]
    return cols


def build_datasets(
    df_tr: pd.DataFrame,
    df_val: pd.DataFrame,
    obs_lt_cols: Sequence[str],
    obs_u_cols: Sequence[str],
    indicator_cols: Sequence[str],
    label_col: str,
    num_choices: int,
    scaler: str = "standard",
    path_col: str = "frames_route",
    audio_cached_col: str | None = "audio_cached_path",
    timestamp_col: str = "timestamp",
    window_id_col: str = "window",
    participant_col: str = "participant",
    audio_start_col: str = "audio_segment_start",
    audio_root: str | None = None,
    video_root: str | None = None,
    audio_norm: str = "per_channel",
    audio_sr: int = 16000,
    audio_duration: float = 5.0,
    audio_template: str = "raw_audio_{participant}.wav",
    audio_fallback_template: str | None = None,
    cat_unique_threshold: int = 5,
):
    # OBS_LT preprocessing (para el encoder multimodal)
    X_lt_tr_mat, preproc_lt, obs_lt_cols = prepare_preprocessor(df_tr, obs_lt_cols, cat_unique_threshold=cat_unique_threshold)
    if len(df_val):
        X_lt_val_mat = preproc_lt.transform(df_val[obs_lt_cols].copy())
    else:
        X_lt_val_mat = np.zeros((0, X_lt_tr_mat.shape[1]), dtype=np.float32)

    # OBS_U preprocessing
    X_u_tr_mat, preproc_u, obs_u_cols = prepare_preprocessor(df_tr, obs_u_cols, cat_unique_threshold=cat_unique_threshold)
    if len(df_val):
        X_u_val_mat = preproc_u.transform(df_val[obs_u_cols].copy())
    else:
        X_u_val_mat = np.zeros((0, X_u_tr_mat.shape[1]), dtype=np.float32)

    # Indicadores
    if indicator_cols and len(df_val):
        ind_tr_mat, preproc_i, indicator_cols = prepare_preprocessor(df_tr, indicator_cols, cat_unique_threshold=cat_unique_threshold)
        ind_val_mat = preproc_i.transform(df_val[indicator_cols].copy())
        ind_tr_mat = to_float_array(ind_tr_mat)
        ind_val_mat = to_float_array(ind_val_mat)
    elif indicator_cols:
        ind_tr_mat, preproc_i, indicator_cols = prepare_preprocessor(df_tr, indicator_cols, cat_unique_threshold=cat_unique_threshold)
        ind_tr_mat = to_float_array(ind_tr_mat)
        ind_val_mat = np.zeros((0, ind_tr_mat.shape[1]), dtype=np.float32)
    else:
        ind_tr_mat = np.zeros((len(df_tr), 0), dtype=np.float32)
        ind_val_mat = np.zeros((len(df_val), 0), dtype=np.float32)

    # Base multimodal dataset (usa OBS_LT como tabular)
    if video_root:
        df_tr = df_tr.copy()
        df_val = df_val.copy()
        df_tr[path_col] = df_tr[path_col].astype(str).apply(lambda p: str(Path(video_root) / p) if not Path(p).is_absolute() else p)
        df_val[path_col] = df_val[path_col].astype(str).apply(lambda p: str(Path(video_root) / p) if not Path(p).is_absolute() else p)
    if audio_cached_col and audio_root:
        df_tr[audio_cached_col] = df_tr[audio_cached_col].astype(str).apply(
            lambda p: str(Path(audio_root) / p) if p not in ("", "nan", "None") and not Path(p).is_absolute() else p
        )
        df_val[audio_cached_col] = df_val[audio_cached_col].astype(str).apply(
            lambda p: str(Path(audio_root) / p) if p not in ("", "nan", "None") and not Path(p).is_absolute() else p
        )

    base_tr = MultimodalAudioDataset(
        df=df_tr,
        tab_columns=obs_lt_cols,
        X_tab_array=torch.tensor(to_float_array(X_lt_tr_mat)),
        path_col=path_col,
        label_col=label_col,
        timestamp_col=timestamp_col,
        window_id_col=window_id_col,
        participant_col=participant_col,
        audio_start_col=audio_start_col,
        audio_cached_col=audio_cached_col,
        audio_root=audio_root,
        audio_template=audio_template,
        audio_fallback_template=audio_fallback_template,
        audio_sr=audio_sr,
        audio_duration=audio_duration,
        audio_norm=audio_norm,
    )
    base_val = MultimodalAudioDataset(
        df=df_val,
        tab_columns=obs_lt_cols,
        X_tab_array=torch.tensor(to_float_array(X_lt_val_mat)),
        path_col=path_col,
        label_col=label_col,
        timestamp_col=timestamp_col,
        window_id_col=window_id_col,
        participant_col=participant_col,
        audio_start_col=audio_start_col,
        audio_cached_col=audio_cached_col,
        audio_root=audio_root,
        audio_template=audio_template,
        audio_fallback_template=audio_fallback_template,
        audio_sr=audio_sr,
        audio_duration=audio_duration,
        audio_norm=audio_norm,
    )

    obs_u_tr_t = torch.tensor(to_float_array(X_u_tr_mat), dtype=torch.float32)
    obs_u_val_t = torch.tensor(to_float_array(X_u_val_mat), dtype=torch.float32)
    ind_tr_t = torch.tensor(ind_tr_mat, dtype=torch.float32)
    ind_val_t = torch.tensor(ind_val_mat, dtype=torch.float32)

    train_ds = MultimodalICLVDataset(base_tr, obs_u_tr_t, ind_tr_t, n_choices=num_choices)
    val_ds = MultimodalICLVDataset(base_val, obs_u_val_t, ind_val_t, n_choices=num_choices)

    return train_ds, val_ds, preproc_lt, preproc_u, preproc_i if indicator_cols else None, obs_lt_cols, obs_u_cols, indicator_cols


def run_epoch(model, loader, device, train: bool = True, optimizer=None, grad_clip: float = 0.0):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = total_choice = total_meas = total_ll = 0.0
    correct = total = 0
    for batch in loader:
        x_tab = batch.x_tab.to(device)
        x_vid = batch.x_vid.to(device)
        x_aud = batch.x_aud.to(device) if batch.x_aud is not None else None
        obs_u = batch.obs_u.to(device)
        indicators = batch.indicators.to(device)
        y = batch.y.to(device)
        if x_vid.dim() == 5:
            x_vid = x_vid[:, :3]  # usar 3 frames
        if x_aud is not None:
            max_len = int(loader.dataset.base.audio_duration * loader.dataset.base.audio_sr) if hasattr(loader.dataset, "base") else None
            if max_len is None:
                max_len = int(16000 * 2)
            x_aud = x_aud[..., :max_len]

        out = model(x_tab, x_vid, x_aud, obs_u, indicators, y)
        loss = out["loss"]
        if train:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        bsz = x_tab.size(0)
        total_loss += float(loss.item()) * bsz
        total_choice += float(out["loss_choice"].item()) * bsz
        total_meas += float(out["loss_meas"].item()) * bsz
        total_ll += float(out["log_likelihood"].item())
        preds = out["logp"].argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += bsz

    return {
        "loss": total_loss / max(1, total),
        "loss_choice": total_choice / max(1, total),
        "loss_meas": total_meas / max(1, total),
        "log_likelihood": total_ll,
        "avg_log_likelihood": total_ll / max(1, total),
        "acc": correct / max(1, total),
    }


def _num_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def eval_loader_metrics(model, loader, device):
    if loader is None:
        return {}
    ys, preds, logps = [], [], []
    total_loglik = 0.0
    n_obs = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x_tab = batch.x_tab.to(device)
            x_vid = batch.x_vid.to(device)
            x_aud = batch.x_aud.to(device) if batch.x_aud is not None else None
            obs_u = batch.obs_u.to(device)
            indicators = batch.indicators.to(device)
            y = batch.y.to(device)
            if x_vid.dim() == 5:
                x_vid = x_vid[:, :3]
            if x_aud is not None:
                max_len = int(loader.dataset.base.audio_duration * loader.dataset.base.audio_sr) if hasattr(loader.dataset, "base") else int(16000 * 2)
                x_aud = x_aud[..., :max_len]
            out = model(x_tab, x_vid, x_aud, obs_u, indicators, y)
            lp = out["logp"].detach().cpu()
            ys.append(y.cpu())
            preds.append(lp.argmax(dim=1))
            logps.append(lp)
            idx = (torch.arange(lp.size(0)), y.cpu())
            total_loglik += float(lp[idx].sum().item())
            n_obs += int(lp.size(0))
    if not ys:
        return {}
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(preds).numpy()
    logp_np = torch.cat(logps).numpy()
    metrics = classification_report_basic(y_true, y_pred, log_probs=logp_np)
    metrics["log_likelihood"] = total_loglik
    metrics["pseudo_r2_mcfadden"] = pseudo_r2_mcfadden(total_loglik, y_true)
    k = _num_params(model)
    metrics["aic"] = float(2 * k - 2 * total_loglik)
    metrics["bic"] = float(np.log(max(1, n_obs)) * k - 2 * total_loglik)
    return metrics


def main():
    ap = argparse.ArgumentParser(description="ICLV con encoder multimodal determinista (tab OBS_LT + video + audio).")
    ap.add_argument("--pkl", type=str, default="data/processed/multimodal_av_join_audio_cached.pkl")
    ap.add_argument("--label-col", type=str, default="action_proc")
    ap.add_argument("--obs-lt-cols-file", type=str, default=None, help="Archivo con OBS_LT")
    ap.add_argument("--obs-u-cols-file", type=str, default=None, help="Archivo con OBS_U")
    ap.add_argument("--indicator-cols-file", type=str, default=None, help="Archivo con OBS_I (indicadores)")
    ap.add_argument("--obs-lt-cols", nargs="*", default=None)
    ap.add_argument("--obs-u-cols", nargs="*", default=None)
    ap.add_argument("--indicator-cols", nargs="*", default=None)
    ap.add_argument("--features-file", type=str, default="utils/feature_sets/exp1.json", help="Fallback si no se dan listas")
    ap.add_argument("--n-latent", type=int, default=64, help="Dim del embedding z (shared_dim)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--delta-shared", action="store_true")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--test-split", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--participant-frac", type=float, default=1.0)
    ap.add_argument("--half-data", action="store_true", help="Usar 50% de participantes para acelerar")
    ap.add_argument("--tabular-scaler", type=str, default="standard", choices=["standard", "robust"])
    # Video/audio paths
    ap.add_argument("--path-col", type=str, default="frames_route")
    ap.add_argument("--video-root", type=str, default=None)
    ap.add_argument("--audio-cached-col", type=str, default="audio_cached_path")
    ap.add_argument("--audio-root", type=str, default=None)
    ap.add_argument("--timestamp-col", type=str, default="timestamp")
    ap.add_argument("--window-id-col", type=str, default="window")
    ap.add_argument("--participant-col", type=str, default="participant")
    ap.add_argument("--audio-start-col", type=str, default="audio_segment_start")
    ap.add_argument("--audio-template", type=str, default="raw_audio_{participant}.wav")
    ap.add_argument("--audio-fallback-template", type=str, default=None)
    ap.add_argument("--audio-sr", type=int, default=16000)
    ap.add_argument("--audio-duration", type=float, default=2.0)
    ap.add_argument("--audio-norm", type=str, default="per_channel", choices=["per_channel", "none"])
    ap.add_argument("--fuse-dropout", type=float, default=0.0)
    ap.add_argument("--freeze-video", action="store_true", help="Congela el encoder de video para acelerar")
    ap.add_argument("--freeze-audio", action="store_true", help="Congela el encoder de audio")
    ap.add_argument("--cat-unique-threshold", type=int, default=5, help="Umbral de unicos para categoricas")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        raise FileNotFoundError(f"No existe el pickle {pkl_path}")
    df = pd.read_pickle(pkl_path).reset_index(drop=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    # Submuestreo de participantes si se solicita
    if 0 < args.participant_frac < 1.0:
        rng = np.random.RandomState(args.seed)
        parts = pd.Index(df[args.participant_col].dropna().unique())
        k = max(1, int(np.ceil(len(parts) * args.participant_frac)))
        keep_parts = rng.choice(parts, size=k, replace=False)
        df = df[df[args.participant_col].isin(keep_parts)].reset_index(drop=True)
        print(f"Subconjunto de participantes: {len(keep_parts)}/{len(parts)} (frac={args.participant_frac})")
    if args.half_data:
        parts = pd.Index(df[args.participant_col].dropna().unique())
        if len(parts) > 0:
            rng = np.random.RandomState(args.seed)
            k = max(1, int(np.ceil(len(parts) * 0.5)))
            keep_parts = rng.choice(parts, size=k, replace=False)
            df = df[df[args.participant_col].isin(keep_parts)].reset_index(drop=True)
            print(f"Subconjunto de participantes: {len(keep_parts)}/{len(parts)} (frac=0.5)")

    # Submuestreo de participantes
    if 0 < args.participant_frac < 1.0:
        rng = np.random.RandomState(args.seed)
        parts = pd.Index(df[args.participant_col].dropna().unique())
        k = max(1, int(np.ceil(len(parts) * args.participant_frac)))
        keep_parts = rng.choice(parts, size=k, replace=False)
        df = df[df[args.participant_col].isin(keep_parts)].reset_index(drop=True)
        print(f"Subconjunto de participantes: {len(keep_parts)}/{len(parts)} (frac={args.participant_frac})")

    drop_cols = {args.label_col, args.path_col, args.audio_cached_col, args.timestamp_col, args.window_id_col, args.participant_col}
    base_features_file = args.features_file
    if args.obs_lt_cols_file or args.obs_u_cols_file or args.indicator_cols_file:
        base_features_file = None

    obs_lt_cols = [c.strip().lower().replace(" ", "_") for c in (args.obs_lt_cols or [])] if args.obs_lt_cols else args.obs_lt_cols
    obs_u_cols = [c.strip().lower().replace(" ", "_") for c in (args.obs_u_cols or [])] if args.obs_u_cols else args.obs_u_cols
    indicator_cols = [c.strip().lower().replace(" ", "_") for c in (args.indicator_cols or [])] if args.indicator_cols else args.indicator_cols

    obs_lt_cols = resolve_cols(df, obs_lt_cols, args.obs_lt_cols_file or base_features_file, drop_cols)
    obs_u_cols = resolve_cols(df, obs_u_cols, args.obs_u_cols_file or base_features_file, drop_cols)
    indicator_cols = resolve_cols(df, indicator_cols, args.indicator_cols_file, set())
    if not indicator_cols:
        indicator_cols = []

    # Label mapping
    default_class_map = {
        "accelerate": 0,
        "brake": 1,
        "decelerate": 2,
        "maintain speed": 3,
        "wait": 4,
    }
    if df[args.label_col].dtype == object:
        df[args.label_col] = df[args.label_col].map(default_class_map)
    df = df.dropna(subset=[args.label_col]).reset_index(drop=True)
    df[args.label_col] = df[args.label_col].astype(int)
    num_choices = int(pd.Series(df[args.label_col]).nunique())

    df_tr, df_val, df_te, info = split_by_participant(
        df,
        participant_col=args.participant_col,
        val_frac=args.val_split,
        test_frac=args.test_split,
        seed=args.seed,
    )
    print(format_split_report(info))

    train_ds, val_ds, preproc_lt, preproc_u, preproc_i, obs_lt_cols, obs_u_cols, indicator_cols = build_datasets(
        df_tr=df_tr,
        df_val=df_val,
        obs_lt_cols=obs_lt_cols,
        obs_u_cols=obs_u_cols,
        indicator_cols=indicator_cols,
        label_col=args.label_col,
        num_choices=num_choices,
        scaler=args.tabular_scaler,
        path_col=args.path_col,
        audio_cached_col=args.audio_cached_col,
        timestamp_col=args.timestamp_col,
        window_id_col=args.window_id_col,
        participant_col=args.participant_col,
        audio_start_col=args.audio_start_col,
        audio_root=args.audio_root,
        video_root=args.video_root,
        audio_norm=args.audio_norm,
        audio_sr=args.audio_sr,
        audio_duration=args.audio_duration,
        audio_template=args.audio_template,
        audio_fallback_template=args.audio_fallback_template,
        cat_unique_threshold=args.cat_unique_threshold,
    )
    # Test dataset opcional
    test_ds = None
    if len(df_te):
        X_lt_te_mat = preproc_lt.transform(df_te[obs_lt_cols].copy())
        X_u_te_mat = preproc_u.transform(df_te[obs_u_cols].copy())
        if indicator_cols and preproc_i is not None:
            ind_te_mat = to_float_array(preproc_i.transform(df_te[indicator_cols].copy()))
        else:
            ind_te_mat = np.zeros((len(df_te), 0), dtype=np.float32)

        base_te = MultimodalAudioDataset(
            df=df_te,
            tab_columns=obs_lt_cols,
            X_tab_array=torch.tensor(to_float_array(X_lt_te_mat)),
            path_col=args.path_col,
            label_col=args.label_col,
            timestamp_col=args.timestamp_col,
            window_id_col=args.window_id_col,
            participant_col=args.participant_col,
            audio_start_col=args.audio_start_col,
            audio_cached_col=args.audio_cached_col,
            audio_root=args.audio_root,
            audio_template=args.audio_template,
            audio_fallback_template=args.audio_fallback_template,
            audio_sr=args.audio_sr,
            audio_duration=args.audio_duration,
            audio_norm=args.audio_norm,
        )
        obs_u_te_t = torch.tensor(to_float_array(X_u_te_mat), dtype=torch.float32)
        ind_te_t = torch.tensor(ind_te_mat, dtype=torch.float32)
        test_ds = MultimodalICLVDataset(base_te, obs_u_te_t, ind_te_t, n_choices=num_choices)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = MultimodalICLVDeterministic(
        tab_in_dim=train_ds.base.X_tab_array.shape[1] if hasattr(train_ds.base, "X_tab_array") else train_ds.base.tab_columns.__len__(),
        dim_obs_u=train_ds.obs_u.shape[-1],
        n_indicators=train_ds.indicators.shape[1],
        n_choices=num_choices,
        shared_dim=args.n_latent,
        alpha=args.alpha,
        delta_per_alt=not args.delta_shared,
        fuse_dropout=args.fuse_dropout,
        freeze_video=args.freeze_video,
        freeze_audio=args.freeze_audio,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_multimodal_icl_v)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multimodal_icl_v)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multimodal_icl_v) if test_ds is not None else None

    history = []
    for epoch in range(1, args.epochs + 1):
        tr_metrics = run_epoch(model, train_loader, device=device, train=True, optimizer=optimizer, grad_clip=args.grad_clip)
        val_metrics = run_epoch(model, val_loader, device=device, train=False, grad_clip=0.0)
        history.append({"epoch": epoch, "train": tr_metrics, "val": val_metrics})
        print(
            f"Epoch {epoch:03d} | "
            f"train loss={tr_metrics['loss']:.4f} acc={tr_metrics['acc']:.3f} ll={tr_metrics['avg_log_likelihood']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.3f} ll={val_metrics['avg_log_likelihood']:.4f}"
        )

    results_dir = Path("results/latest_iclv")
    ensure_dir(results_dir)
    base_config = {
        "pkl": str(pkl_path),
        "label_col": args.label_col,
        "obs_lt_cols": list(obs_lt_cols),
        "obs_u_cols": list(obs_u_cols),
        "indicator_cols": list(indicator_cols),
        "obs_lt_cols_file": args.obs_lt_cols_file,
        "obs_u_cols_file": args.obs_u_cols_file,
        "indicator_cols_file": args.indicator_cols_file,
        "n_latent": args.n_latent,
        "alpha": args.alpha,
        "delta_shared": args.delta_shared,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "tabular_scaler": args.tabular_scaler,
        "cat_unique_threshold": args.cat_unique_threshold,
        "seed": args.seed,
        "device": str(device),
        "freeze_video": args.freeze_video,
        "freeze_audio": args.freeze_audio,
        "grad_clip": args.grad_clip,
        "participant_col": args.participant_col,
        "participant_frac": args.participant_frac,
        "half_data": args.half_data,
        "argv": sys.argv,
    }
    run_hash = compute_run_hash(base_config, sys.argv, model="MM_ICLV")

    metrics_val = eval_loader_metrics(model, val_loader, device=device)
    metrics_test = eval_loader_metrics(model, test_loader, device=device) if test_loader is not None else {}
    all_metrics = {f"val_{k}": v for k, v in metrics_val.items()}
    all_metrics.update({f"test_{k}": v for k, v in metrics_test.items()})
    save_metrics(all_metrics, results_dir, model_name="MM_ICLV", config=base_config, run_hash=run_hash)

    split_path = results_dir / "MM_ICLV" / "split_info.txt"
    split_path.write_text(format_split_report(info), encoding="utf-8")

    model_path = results_dir / artifact_name("MM_ICLV", "model", run_hash, "pt")
    preproc_lt_path = results_dir / artifact_name("MM_ICLV", "preproc_lt", run_hash, "pkl")
    preproc_u_path = results_dir / artifact_name("MM_ICLV", "preproc_u", run_hash, "pkl")
    hist_path = results_dir / artifact_name("MM_ICLV", "history", run_hash, "json")
    metrics_path = results_dir / "MM_ICLV" / f"metrics_{run_hash}.json"

    torch.save(model.state_dict(), model_path)
    save_model_pickle(preproc_lt, preproc_lt_path)
    save_model_pickle(preproc_u, preproc_u_path)
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    register_run(results_dir, run_hash, "MM_ICLV", cmd=" ".join(sys.argv), config=base_config)

    print(f"[OK] run_hash={run_hash}")
    print(f"[OK] model: {model_path}")
    print(f"[OK] preproc_lt: {preproc_lt_path}")
    print(f"[OK] preproc_u: {preproc_u_path}")
    print(f"[OK] history: {hist_path}")
    if metrics_path.exists():
        print(f"[OK] metrics: {metrics_path}")


if __name__ == "__main__":
    main()
