from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

# Ensure package root on path (dataset_bicicletas/)
ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loading.icl_v import ICLVDataset
from src.data_loading.multimodal_audio import MultimodalAudioDataset
from src.data_loading.multimodal_icl_v import MultimodalICLVDataset, collate_multimodal_icl_v
from src.models.icl_v import DeterministicICLV, MultimodalICLVDeterministic
from src.data_cleaning.cleaning import categorias_a_str, convertir_a_categorico
from utils.features import load_features_file


def _load_pickle(path: Path):
    return joblib.load(path)


def _to_float_array(mat) -> np.ndarray:
    try:
        arr = mat.toarray()
    except Exception:
        arr = np.asarray(mat)
    return arr.astype(np.float32)


def _encode_indicators(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    blocks = []
    for col in cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            col_vals = df[col].fillna(df[col].median())
        else:
            col_str = df[col].astype(str)
            uniq = col_str.unique().tolist()
            mapping = {v: i for i, v in enumerate(uniq)}
            col_vals = col_str.map(mapping).fillna(-1)
        blocks.append(col_vals.to_numpy(dtype=np.float32))
    if not blocks:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.stack(blocks, axis=1).astype(np.float32)


def _target_params(model: torch.nn.Module, include: Iterable[str]) -> List[Tuple[str, torch.nn.Parameter]]:
    out: List[Tuple[str, torch.nn.Parameter]] = []
    include = set(s.strip().lower() for s in include)
    if "beta" in include and hasattr(model, "beta"):
        out.append(("beta", model.beta))
    if "asc" in include and hasattr(model, "ASC"):
        out.append(("ASC", model.ASC))
    if "delta" in include and hasattr(model, "delta"):
        out.append(("delta", model.delta))
    return out


def _expand_param_names(named_params: List[Tuple[str, torch.nn.Parameter]]) -> List[str]:
    names: List[str] = []
    for n, p in named_params:
        if p.numel() == 1:
            names.append(n)
        else:
            names.extend([f"{n}[{i}]" for i in range(p.numel())])
    return names


def _opg_from_loader(model, loader, params, device, max_batches: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    params_list = [p for _, p in params]
    flat_theta = torch.nn.utils.parameters_to_vector(params_list).detach().to(device)
    dim = flat_theta.numel()
    opg = torch.zeros((dim, dim), device=device)
    n = 0

    for b_idx, batch in enumerate(loader):
        if max_batches and b_idx >= max_batches:
            break
        if isinstance(batch, tuple):
            obs_lt, obs_u, indicators, choice = batch
            obs_lt = obs_lt.to(device)
            obs_u = obs_u.to(device)
            indicators = indicators.to(device)
            choice = torch.as_tensor(choice, device=device, dtype=torch.long)
            out = model(obs_lt, obs_u, indicators, choice)
        else:
            x_tab = batch.x_tab.to(device)
            x_vid = batch.x_vid.to(device)
            x_aud = batch.x_aud.to(device) if batch.x_aud is not None else None
            obs_u = batch.obs_u.to(device)
            indicators = batch.indicators.to(device)
            choice = batch.y.to(device)
            out = model(x_tab, x_vid, x_aud, obs_u, indicators, choice)

        logp = out["logp"]
        for i in range(logp.size(0)):
            loglik_i = logp[i, choice[i]]
            grads = torch.autograd.grad(
                loglik_i,
                params_list,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            flat_grads = []
            for g, (_, p) in zip(grads, params):
                if g is None:
                    flat_grads.append(torch.zeros(p.numel(), device=device))
                else:
                    flat_grads.append(g.reshape(-1))
            g_vec = torch.cat(flat_grads)
            opg += torch.outer(g_vec, g_vec)
            n += 1

    if n == 0:
        raise RuntimeError("No se encontraron muestras para OPG.")
    info = opg / float(n)
    eye = torch.eye(info.shape[0], device=info.device, dtype=info.dtype) * 1e-6
    info_safe = info + eye
    cov = torch.linalg.pinv(info_safe)
    std = torch.sqrt(torch.clamp(torch.diag(cov), min=1e-12))
    tstat = flat_theta / torch.clamp(std, min=1e-12)
    return flat_theta.detach().cpu(), std.detach().cpu(), tstat.detach().cpu()


def _build_iclv_model(state: dict) -> DeterministicICLV:
    beta = state["beta"]
    gamma_w = state["Gamma.weight"]
    dim_obs_lt = int(gamma_w.shape[1])
    n_latent = int(gamma_w.shape[0])
    dim_obs_u = int(beta.shape[1])
    n_choices = int(beta.shape[0])
    n_indicators = int(state["Lambda.weight"].shape[0]) if "Lambda.weight" in state else 0
    delta_per_alt = state["delta"].dim() == 2
    model = DeterministicICLV(
        dim_obs_lt=dim_obs_lt,
        dim_obs_u=dim_obs_u,
        n_latent=n_latent,
        n_indicators=n_indicators,
        n_choices=n_choices,
        alpha=0.0,
        delta_per_alt=delta_per_alt,
    )
    model.load_state_dict(state, strict=True)
    return model


def _build_mm_iclv_model(state: dict, tab_in_dim: int, dim_obs_u: int, n_indicators: int) -> MultimodalICLVDeterministic:
    beta = state["beta"]
    n_choices = int(beta.shape[0])
    shared_dim = int(state["delta"].shape[1]) if state["delta"].dim() == 2 else int(state["delta"].shape[0])
    delta_per_alt = state["delta"].dim() == 2
    tab_emb_dim = int(state["tab_enc.net.0.weight"].shape[0]) if "tab_enc.net.0.weight" in state else 128
    model = MultimodalICLVDeterministic(
        tab_in_dim=tab_in_dim,
        dim_obs_u=dim_obs_u,
        n_indicators=n_indicators,
        n_choices=n_choices,
        tab_emb_dim=tab_emb_dim,
        shared_dim=shared_dim,
        alpha=0.0,
        delta_per_alt=delta_per_alt,
        fuse_dropout=0.0,
        freeze_video=True,
        freeze_audio=True,
    )
    model.load_state_dict(state, strict=False)
    return model


def run_iclv(args):
    state = torch.load(args.iclv_model_pt, map_location="cpu", weights_only=True)
    model = _build_iclv_model(state)
    params = _target_params(model, args.iclv_params.split(","))
    names = _expand_param_names(params)

    pkl_path = Path(args.iclv_pkl)
    df = pd.read_pickle(pkl_path) if pkl_path.suffix.lower() != ".csv" else pd.read_csv(pkl_path, low_memory=False)
    df = df.reset_index(drop=True)

    obs_lt_cols = load_features_file(args.iclv_obs_lt_cols_file) if args.iclv_obs_lt_cols_file else []
    obs_u_cols = load_features_file(args.iclv_obs_u_cols_file) if args.iclv_obs_u_cols_file else []
    ind_cols = load_features_file(args.iclv_indicator_cols_file) if args.iclv_indicator_cols_file else []

    df = df.dropna(subset=[args.iclv_label_col]).reset_index(drop=True)
    if df[args.iclv_label_col].dtype == object:
        default_class_map = {
            "accelerate": 0,
            "brake": 1,
            "decelerate": 2,
            "maintain speed": 3,
            "wait": 4,
        }
        df[args.iclv_label_col] = df[args.iclv_label_col].map(default_class_map)
    df = df.dropna(subset=[args.iclv_label_col]).reset_index(drop=True)
    df[args.iclv_label_col] = df[args.iclv_label_col].astype(int)

    preproc_lt = _load_pickle(Path(args.iclv_preproc_lt)) if args.iclv_preproc_lt else None
    preproc_u = _load_pickle(Path(args.iclv_preproc_u)) if args.iclv_preproc_u else None
    if preproc_lt is None or preproc_u is None:
        raise ValueError("Debe proveer --iclv-preproc-lt y --iclv-preproc-u para OPG consistente.")

    X_lt = _to_float_array(preproc_lt.transform(convertir_a_categorico(categorias_a_str(df[obs_lt_cols].copy()))))
    X_u = _to_float_array(preproc_u.transform(convertir_a_categorico(categorias_a_str(df[obs_u_cols].copy()))))
    indicators = _encode_indicators(df, ind_cols)

    y = df[args.iclv_label_col].to_numpy(dtype=np.int64)
    num_choices = int(pd.Series(y).nunique())
    ds = ICLVDataset(obs_lt=X_lt, obs_u=X_u, indicators=indicators, choices=y, num_choices=num_choices)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.iclv_batch_size, shuffle=False)

    device = torch.device(args.device)
    model = model.to(device)
    theta, std, tstat = _opg_from_loader(model, loader, params, device, max_batches=args.iclv_max_batches)

    out_path = Path(args.iclv_out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"name": names, "theta": theta.numpy(), "std": std.numpy(), "tstat": tstat.numpy()}).to_csv(out_path, index=False)
    print(f"[OK] ICLV OPG guardado en: {out_path}")


def run_mm_iclv(args):
    state = torch.load(args.mm_model_pt, map_location="cpu", weights_only=True)

    pkl_path = Path(args.mm_pkl)
    df = pd.read_pickle(pkl_path).reset_index(drop=True)
    obs_lt_cols = load_features_file(args.mm_obs_lt_file) if args.mm_obs_lt_file else []
    obs_u_cols = load_features_file(args.mm_obs_u_file) if args.mm_obs_u_file else []
    ind_cols = load_features_file(args.mm_indicator_file) if args.mm_indicator_file else []

    df = df.dropna(subset=[args.mm_label_col]).reset_index(drop=True)
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

    preproc_lt = _load_pickle(Path(args.mm_preproc_lt)) if args.mm_preproc_lt else None
    preproc_u = _load_pickle(Path(args.mm_preproc_u)) if args.mm_preproc_u else None
    if preproc_lt is None or preproc_u is None:
        raise ValueError("Debe proveer --mm-preproc-lt y --mm-preproc-u para OPG consistente.")

    X_lt = _to_float_array(preproc_lt.transform(convertir_a_categorico(categorias_a_str(df[obs_lt_cols].copy()))))
    X_u = _to_float_array(preproc_u.transform(convertir_a_categorico(categorias_a_str(df[obs_u_cols].copy()))))
    indicators = _encode_indicators(df, ind_cols)

    num_choices = int(pd.Series(df[args.mm_label_col]).nunique())
    base = MultimodalAudioDataset(
        df=df,
        tab_columns=obs_lt_cols,
        X_tab_array=torch.tensor(X_lt),
        path_col=args.mm_path_col,
        label_col=args.mm_label_col,
        timestamp_col=args.mm_timestamp_col,
        window_id_col=args.mm_window_id_col,
        participant_col=args.mm_participant_col,
        audio_start_col=args.mm_audio_start_col,
        audio_cached_col=args.mm_audio_cached_col,
        audio_root=args.mm_audio_root,
        audio_template=args.mm_audio_template,
        audio_fallback_template=args.mm_audio_fallback_template,
        audio_sr=args.mm_audio_sr,
        audio_duration=args.mm_audio_duration,
        audio_norm=args.mm_audio_norm,
    )
    obs_u_t = torch.tensor(X_u, dtype=torch.float32)
    ind_t = torch.tensor(indicators, dtype=torch.float32)
    ds = MultimodalICLVDataset(base, obs_u_t, ind_t, n_choices=num_choices)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.mm_batch_size, shuffle=False, collate_fn=collate_multimodal_icl_v)

    model = _build_mm_iclv_model(state, tab_in_dim=X_lt.shape[1], dim_obs_u=X_u.shape[1], n_indicators=indicators.shape[1])
    params = _target_params(model, args.mm_params.split(","))
    names = _expand_param_names(params)

    device = torch.device(args.device)
    model = model.to(device)
    theta, std, tstat = _opg_from_loader(model, loader, params, device, max_batches=args.mm_max_batches)

    out_path = Path(args.mm_out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"name": names, "theta": theta.numpy(), "std": std.numpy(), "tstat": tstat.numpy()}).to_csv(out_path, index=False)
    print(f"[OK] MM_ICLV OPG guardado en: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OPG/Fisher Information para ICLV y MM-ICLV (utility params).")
    ap.add_argument("--device", type=str, default="cpu")
    # ICLV
    ap.add_argument("--iclv-model-pt", type=str, default=None)
    ap.add_argument("--iclv-pkl", type=str, default="data/processed/multimodal_av_join_audio_with_iclv.pkl")
    ap.add_argument("--iclv-obs-lt-cols-file", type=str, default=None)
    ap.add_argument("--iclv-obs-u-cols-file", type=str, default=None)
    ap.add_argument("--iclv-indicator-cols-file", type=str, default=None)
    ap.add_argument("--iclv-label-col", type=str, default="action_proc")
    ap.add_argument("--iclv-preproc-lt", type=str, default=None)
    ap.add_argument("--iclv-preproc-u", type=str, default=None)
    ap.add_argument("--iclv-params", type=str, default="beta,ASC,delta")
    ap.add_argument("--iclv-batch-size", type=int, default=64)
    ap.add_argument("--iclv-max-batches", type=int, default=0)
    ap.add_argument("--iclv-out-csv", type=str, default="results/ICLV/utility_opg.csv")
    # MM ICLV
    ap.add_argument("--mm-model-pt", type=str, default=None)
    ap.add_argument("--mm-pkl", type=str, default="data/processed/multimodal_av_join_audio_with_iclv.pkl")
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
    ap.add_argument("--mm-audio-template", type=str, default="raw_audio_{participant}.wav")
    ap.add_argument("--mm-audio-fallback-template", type=str, default=None)
    ap.add_argument("--mm-audio-sr", type=int, default=16000)
    ap.add_argument("--mm-audio-duration", type=float, default=2.0)
    ap.add_argument("--mm-audio-norm", type=str, default="per_channel")
    ap.add_argument("--mm-preproc-lt", type=str, default=None)
    ap.add_argument("--mm-preproc-u", type=str, default=None)
    ap.add_argument("--mm-params", type=str, default="beta,ASC,delta")
    ap.add_argument("--mm-batch-size", type=int, default=16)
    ap.add_argument("--mm-max-batches", type=int, default=0)
    ap.add_argument("--mm-out-csv", type=str, default="results/MM_ICLV/utility_opg.csv")
    args = ap.parse_args()

    if args.iclv_model_pt:
        run_iclv(args)
    if args.mm_model_pt:
        run_mm_iclv(args)
