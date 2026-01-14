from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

# Ensure package root on path (dataset_bicicletas/)
ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.features import load_features_file
from utils.splits import split_by_participant
from src.data_loading.multimodal_icl_v import collate_multimodal_icl_v
from src.models.icl_v import DeterministicICLV, MultimodalICLVDeterministic, compute_hessian_stats
from mains.run_multimodal_icl_v import build_datasets, resolve_cols


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_params_csv(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    if "name" not in df.columns:
        df = df.rename(columns={df.columns[0]: "name"})
    out = {}
    for _, row in df.iterrows():
        name = str(row["name"]) if "name" in row else None
        if not name:
            continue
        out[name] = {
            "theta": float(row["theta"]) if "theta" in row else float("nan"),
            "std": float(row["std"]) if "std" in row else float("nan"),
            "tstat": float(row["tstat"]) if "tstat" in row else float("nan"),
        }
    return out


def _feature_names_from_preproc(preproc, fallback: List[str]) -> List[str]:
    if preproc is None:
        return fallback
    try:
        names = list(preproc.get_feature_names_out())
        return names
    except Exception:
        return fallback


def _beta_rows(beta: torch.Tensor, asc: torch.Tensor, feature_names: List[str], params_map: Dict[str, Dict[str, float]], model_name: str):
    rows = []
    beta_np = beta.detach().cpu().numpy()
    asc_np = asc.detach().cpu().numpy()
    J, K = beta_np.shape
    if len(feature_names) != K:
        feature_names = feature_names[:K] + [f"obs_u_{i}" for i in range(len(feature_names), K)]
    for j in range(J):
        for k in range(K):
            idx = j * K + k
            name = f"beta[{idx}]"
            stats = params_map.get(name, {})
            rows.append({
                "model": model_name,
                "param_type": "beta",
                "alt": j,
                "feature": feature_names[k],
                "beta": float(beta_np[j, k]),
                "std": float(stats.get("std", float("nan"))),
                "tstat": float(stats.get("tstat", float("nan"))),
            })
    for j in range(len(asc_np)):
        name = f"ASC[{j}]"
        stats = params_map.get(name, {})
        rows.append({
            "model": model_name,
            "param_type": "ASC",
            "alt": j,
            "feature": "ASC",
            "beta": float(asc_np[j]),
            "std": float(stats.get("std", float("nan"))),
            "tstat": float(stats.get("tstat", float("nan"))),
        })
    return rows


def _infer_tab_emb_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    keys = [k for k in state_dict.keys() if k.startswith("tab_enc.net") and k.endswith("weight")]
    if not keys:
        return 128
    def _idx(k):
        try:
            return int(k.split(".")[2])
        except Exception:
            return 0
    last_key = sorted(keys, key=_idx)[-1]
    return int(state_dict[last_key].shape[0])


def extract_iclv_classic(args):
    model_path = Path(args.iclv_model_pt) if args.iclv_model_pt else None
    if model_path is None or not model_path.exists():
        print("[WARN] iclv_model_pt no existe; se omite ICLV clasico")
        return
    state = torch.load(model_path, map_location="cpu")
    beta = state["beta"]
    asc = state["ASC"]

    obs_u_cols = load_features_file(args.iclv_obs_u_file) if args.iclv_obs_u_file else []
    preproc = _load_pickle(Path(args.iclv_preproc_u)) if args.iclv_preproc_u else None
    feature_names = _feature_names_from_preproc(preproc, obs_u_cols)
    params_map = _load_params_csv(Path(args.iclv_params_csv)) if args.iclv_params_csv else {}

    rows = _beta_rows(beta, asc, feature_names, params_map, model_name="ICLV")
    out_path = Path(args.iclv_out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[OK] ICLV: guardado {out_path}")


def extract_mm_iclv(args):
    model_path = Path(args.mm_model_pt) if args.mm_model_pt else None
    if model_path is None or not model_path.exists():
        print("[WARN] mm_model_pt no existe; se omite MM_ICLV")
        return
    state = torch.load(model_path, map_location="cpu")
    beta = state["beta"]
    asc = state["ASC"]

    obs_u_cols = load_features_file(args.mm_obs_u_file) if args.mm_obs_u_file else []

    # reconstruir datasets para hessiano
    pkl_path = Path(args.mm_pkl)
    df = pd.read_pickle(pkl_path).reset_index(drop=True)
    drop_cols = {args.mm_label_col, args.mm_path_col, args.mm_audio_cached_col, args.mm_timestamp_col, args.mm_window_id_col, args.mm_participant_col}
    obs_lt_cols = resolve_cols(df, None, args.mm_obs_lt_file, drop_cols)
    obs_u_cols = resolve_cols(df, None, args.mm_obs_u_file, drop_cols)
    indicator_cols = resolve_cols(df, None, args.mm_indicator_file, set())
    if not indicator_cols:
        indicator_cols = []

    # label mapping (igual que main)
    if df[args.mm_label_col].dtype == object:
        default_class_map = {
            "accelerate": 0,
            "brake": 1,
            "decelerate": 2,
            "maintain speed": 3,
            "wait": 4,
        }
        df[args.mm_label_col] = df[args.mm_label_col].map(default_class_map)
    df = df.dropna(subset=[args.mm_label_col]).reset_index(drop=True)
    df[args.mm_label_col] = df[args.mm_label_col].astype(int)
    num_choices = int(pd.Series(df[args.mm_label_col]).nunique())

    df_tr, df_val, _, _ = split_by_participant(
        df,
        participant_col=args.mm_participant_col,
        val_frac=args.mm_val_split,
        test_frac=0.0,
        seed=args.mm_seed,
    )

    train_ds, _, preproc_lt, preproc_u = build_datasets(
        df_tr=df_tr,
        df_val=df_val,
        obs_lt_cols=obs_lt_cols,
        obs_u_cols=obs_u_cols,
        indicator_cols=indicator_cols,
        label_col=args.mm_label_col,
        num_choices=num_choices,
        scaler=args.mm_tabular_scaler,
        path_col=args.mm_path_col,
        audio_cached_col=args.mm_audio_cached_col,
        timestamp_col=args.mm_timestamp_col,
        window_id_col=args.mm_window_id_col,
        participant_col=args.mm_participant_col,
        audio_start_col=args.mm_audio_start_col,
        audio_root=args.mm_audio_root,
        video_root=args.mm_video_root,
        audio_norm=args.mm_audio_norm,
        audio_sr=args.mm_audio_sr,
        audio_duration=args.mm_audio_duration,
        audio_template=args.mm_audio_template,
        audio_fallback_template=args.mm_audio_fallback_template,
    )

    tab_in_dim = train_ds.base.X_tab_array.shape[1]
    dim_obs_u = train_ds.obs_u.shape[-1]
    n_indicators = train_ds.indicators.shape[1]
    shared_dim = state["delta"].shape[1] if state["delta"].dim() == 2 else state["delta"].shape[0]
    tab_emb_dim = _infer_tab_emb_dim(state)
    delta_per_alt = state["delta"].dim() == 2

    model = MultimodalICLVDeterministic(
        tab_in_dim=tab_in_dim,
        dim_obs_u=dim_obs_u,
        n_indicators=n_indicators,
        n_choices=num_choices,
        tab_emb_dim=tab_emb_dim,
        shared_dim=shared_dim,
        alpha=args.mm_alpha,
        delta_per_alt=delta_per_alt,
        fuse_dropout=0.0,
        freeze_video=args.mm_freeze_video,
        freeze_audio=args.mm_freeze_audio,
    )
    model.load_state_dict(state, strict=False)

    # congelar todo menos beta/delta/ASC
    for name, p in model.named_parameters():
        if not (name.startswith("beta") or name.startswith("delta") or name.startswith("ASC")):
            p.requires_grad = False

    device = torch.device(args.mm_device)
    model = model.to(device)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.mm_batch_size,
        shuffle=False,
        collate_fn=collate_multimodal_icl_v,
    )

    def loss_closure():
        model.eval()
        total = 0.0
        n = 0
        for i, batch in enumerate(train_loader):
            if args.mm_max_batches and i >= args.mm_max_batches:
                break
            x_tab = batch.x_tab.to(device)
            x_vid = batch.x_vid.to(device)
            x_aud = batch.x_aud.to(device) if batch.x_aud is not None else None
            obs_u = batch.obs_u.to(device)
            indicators = batch.indicators.to(device)
            y = batch.y.to(device)
            if x_vid.dim() == 5:
                x_vid = x_vid[:, :3]
            if x_aud is not None:
                max_len = int(args.mm_audio_sr * args.mm_audio_duration)
                x_aud = x_aud[..., :max_len]
            out = model(x_tab, x_vid, x_aud, obs_u, indicators, y)
            total = total + out["loss"]
            n += 1
        return total / max(1, n)

    hess = compute_hessian_stats(model, loss_closure)
    params_map = {name: {"theta": float(t), "std": float(s), "tstat": float(tt)} for name, t, s, tt in zip(hess.names, hess.theta, hess.std, hess.tstat)}

    feature_names = _feature_names_from_preproc(preproc_u, obs_u_cols)
    rows = _beta_rows(beta, asc, feature_names, params_map, model_name="MM_ICLV")
    out_path = Path(args.mm_out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[OK] MM_ICLV: guardado {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extrae betas/ASC con std y t-stat para ICLV y MM_ICLV")
    # ICLV clasico
    ap.add_argument("--iclv-model-pt", type=str, default=None)
    ap.add_argument("--iclv-params-csv", type=str, default=None)
    ap.add_argument("--iclv-preproc-u", type=str, default=None)
    ap.add_argument("--iclv-obs-u-file", type=str, default=None)
    ap.add_argument("--iclv-out-csv", type=str, default="results/ICLV/utility_stats.csv")
    # MM ICLV
    ap.add_argument("--mm-model-pt", type=str, default=None)
    ap.add_argument("--mm-pkl", type=str, default="data/processed/multimodal_av_join_audio_cached.pkl")
    ap.add_argument("--mm-obs-lt-file", type=str, default=None)
    ap.add_argument("--mm-obs-u-file", type=str, default=None)
    ap.add_argument("--mm-indicator-file", type=str, default=None)
    ap.add_argument("--mm-label-col", type=str, default="action_proc")
    ap.add_argument("--mm-path-col", type=str, default="frames_route")
    ap.add_argument("--mm-audio-cached-col", type=str, default="audio_cached_path")
    ap.add_argument("--mm-timestamp-col", type=str, default="timestamp")
    ap.add_argument("--mm-window-id-col", type=str, default="window")
    ap.add_argument("--mm-participant-col", type=str, default="participant")
    ap.add_argument("--mm-audio-start-col", type=str, default="audio_segment_start")
    ap.add_argument("--mm-audio-root", type=str, default=None)
    ap.add_argument("--mm-video-root", type=str, default=None)
    ap.add_argument("--mm-audio-norm", type=str, default="per_channel")
    ap.add_argument("--mm-audio-sr", type=int, default=16000)
    ap.add_argument("--mm-audio-duration", type=float, default=2.0)
    ap.add_argument("--mm-audio-template", type=str, default="raw_audio_{participant}.wav")
    ap.add_argument("--mm-audio-fallback-template", type=str, default=None)
    ap.add_argument("--mm-tabular-scaler", type=str, default="standard")
    ap.add_argument("--mm-alpha", type=float, default=1.0)
    ap.add_argument("--mm-device", type=str, default="cpu")
    ap.add_argument("--mm-freeze-video", action="store_true", default=True)
    ap.add_argument("--mm-freeze-audio", action="store_true", default=True)
    ap.add_argument("--mm-batch-size", type=int, default=16)
    ap.add_argument("--mm-max-batches", type=int, default=10, help="0 para usar todo")
    ap.add_argument("--mm-val-split", type=float, default=0.2)
    ap.add_argument("--mm-seed", type=int, default=42)
    ap.add_argument("--mm-out-csv", type=str, default="results/MM_ICLV/utility_stats.csv")

    args = ap.parse_args()

    extract_iclv_classic(args)
    extract_mm_iclv(args)
