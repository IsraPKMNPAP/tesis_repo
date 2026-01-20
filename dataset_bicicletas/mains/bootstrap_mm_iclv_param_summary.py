from __future__ import annotations

import argparse
import json
import time
import re
from pathlib import Path
from typing import List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

# Ensure package root on path (dataset_bicicletas/)
ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loading.multimodal_audio import MultimodalAudioDataset
from src.data_loading.multimodal_icl_v import MultimodalICLVDataset, collate_multimodal_icl_v
from src.models.icl_v import MultimodalICLVDeterministic
from src.data_cleaning.cleaning import categorias_a_str, convertir_a_categorico
from utils.features import load_features_file


def parse_run_index(idx_path: Path) -> Tuple[str, dict]:
    if not idx_path.exists():
        raise FileNotFoundError(f"No existe {idx_path}")
    blocks = [b.strip() for b in idx_path.read_text(encoding="utf-8").split("-----") if b.strip()]
    for block in reversed(blocks):
        if "model=MM_ICLV" not in block:
            continue
        m_hash = re.search(r"hash=(\w+)", block)
        m_cfg = re.search(r"config:\s*(\{.*\})", block, flags=re.S)
        if not m_hash or not m_cfg:
            continue
        run_hash = m_hash.group(1)
        config = json.loads(m_cfg.group(1))
        return run_hash, config
    raise ValueError("No se encontro un run MM_ICLV en run_index.txt")


def parse_train_parts(split_path: Path) -> List[str]:
    if not split_path.exists():
        return []
    text = split_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("Train:") and "->" in line:
            tail = line.split("->", 1)[1].strip()
            try:
                parts = json.loads(tail.replace("'", "\""))
                if isinstance(parts, list):
                    return [str(p) for p in parts]
            except Exception:
                try:
                    import ast

                    parts = ast.literal_eval(tail)
                    if isinstance(parts, list):
                        return [str(p) for p in parts]
                except Exception:
                    return []
    return []


def to_float_array(mat) -> np.ndarray:
    try:
        arr = mat.toarray()
    except Exception:
        arr = np.asarray(mat)
    return arr.astype(np.float32)


def encode_indicator_blocks(df_tr: pd.DataFrame, df_full: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    blocks = []
    for col in cols:
        if col not in df_full.columns:
            continue
        if pd.api.types.is_numeric_dtype(df_tr[col]):
            med = df_tr[col].median()
            col_full = df_full[col].fillna(med)
        else:
            tr_str = df_tr[col].astype(str)
            uniq = tr_str.unique().tolist()
            mapping = {v: i for i, v in enumerate(uniq)}
            col_full = df_full[col].astype(str).map(mapping).fillna(-1)
        blocks.append(col_full.to_numpy(dtype=np.float32))
    if not blocks:
        return np.zeros((len(df_full), 0), dtype=np.float32)
    return np.stack(blocks, axis=1).astype(np.float32)


def bootstrap_indices(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(n), size=n, replace=True)


def summarize_params(samples: np.ndarray, names: List[str]) -> pd.DataFrame:
    mean = np.nanmean(samples, axis=0)
    median = np.nanmedian(samples, axis=0)
    mean_abs = np.nanmean(np.abs(samples), axis=0)
    median_abs = np.nanmedian(np.abs(samples), axis=0)
    p25 = np.nanpercentile(samples, 25, axis=0)
    p75 = np.nanpercentile(samples, 75, axis=0)
    sign = np.sign(median)
    sign_pct = np.nanmean(np.sign(samples) == np.sign(median), axis=0)
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "name": name,
                "mean": float(mean[i]),
                "median": float(median[i]),
                "mean_abs": float(mean_abs[i]),
                "median_abs": float(median_abs[i]),
                "p25": float(p25[i]),
                "p75": float(p75[i]),
                "sign": float(sign[i]),
                "sign_pct": float(sign_pct[i]),
            }
        )
    return pd.DataFrame(rows)


def expand_param_names(model: torch.nn.Module) -> tuple[List[torch.nn.Parameter], List[str]]:
    params = []
    names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        if param.numel() == 1:
            names.append(name)
        else:
            names.extend([f"{name}[{i}]" for i in range(param.numel())])
    return params, names


def _count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def parse_flat_index(name: str) -> int | None:
    if "[" not in name or "]" not in name:
        return None
    try:
        idx = name.rsplit("[", 1)[-1].split("]")[0]
        return int(idx)
    except Exception:
        return None


def build_param_annotations(
    names: List[str],
    u_names: List[str],
    indicator_cols: List[str],
    n_choices: int,
) -> pd.DataFrame:
    base_alt = 0
    alt_list = [a for a in range(n_choices) if a != base_alt]
    rows = []
    for name in names:
        block = "other"
        var_name = name
        alt = None
        if name.startswith("beta_shared"):
            block = "utility"
            idx = parse_flat_index(name)
            if idx is not None and u_names:
                var_name = u_names[idx] if idx < len(u_names) else var_name
        elif name.startswith("beta"):
            block = "utility"
            idx = parse_flat_index(name)
            if idx is not None and u_names:
                dim_u = len(u_names)
                alt_pos = idx // dim_u
                feat_idx = idx % dim_u
                if alt_pos < len(alt_list):
                    alt = alt_list[alt_pos]
                var_name = u_names[feat_idx] if feat_idx < len(u_names) else var_name
        elif name.startswith("ASC"):
            block = "utility"
        elif name.startswith("delta"):
            block = "utility"
        elif name.startswith("Lambda.weight"):
            block = "measurement"
            idx = parse_flat_index(name)
            if idx is not None and indicator_cols:
                n_lat = None
                if len(indicator_cols) > 0:
                    n_lat = idx % max(1, len(indicator_cols))
                var_name = indicator_cols[idx // max(1, len(indicator_cols))] if indicator_cols else var_name
        rows.append({"name": name, "block": block, "var_name": var_name, "alt": alt})
    return pd.DataFrame(rows)


def _epoch_pass(model, loader, device, optimizer=None) -> float:
    total_ll = 0.0
    for batch in loader:
        x_tab = batch.x_tab.to(device)
        x_vid = batch.x_vid.to(device)
        x_aud = batch.x_aud.to(device) if batch.x_aud is not None else None
        obs_u = batch.obs_u.to(device)
        indicators = batch.indicators.to(device)
        y = batch.y.to(device)
        out = model(x_tab, x_vid, x_aud, obs_u, indicators, y)
        loss = out["loss"]
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_ll += float(out["log_likelihood"].item())
    return total_ll


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap multimodal: resumen de coeficientes (dataset_bicicletas).")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--run-hash", type=str, default=None)
    ap.add_argument("--pkl", type=Path, default=None)
    ap.add_argument("--label-col", type=str, default=None)
    ap.add_argument("--obs-lt-cols-file", type=str, default=None)
    ap.add_argument("--obs-u-cols-file", type=str, default=None)
    ap.add_argument("--indicator-cols-file", type=str, default=None)
    ap.add_argument("--path-col", type=str, default="frames_route")
    ap.add_argument("--video-root", type=str, default=None)
    ap.add_argument("--audio-cached-col", type=str, default="audio_cached_path")
    ap.add_argument("--audio-root", type=str, default=None)
    ap.add_argument("--timestamp-col", type=str, default="timestamp")
    ap.add_argument("--window-id-col", type=str, default="window")
    ap.add_argument("--participant-col", type=str, default=None)
    ap.add_argument("--audio-start-col", type=str, default="audio_segment_start")
    ap.add_argument("--audio-template", type=str, default="raw_audio_{participant}.wav")
    ap.add_argument("--audio-fallback-template", type=str, default=None)
    ap.add_argument("--audio-sr", type=int, default=16000)
    ap.add_argument("--audio-duration", type=float, default=2.0)
    ap.add_argument("--audio-norm", type=str, default="per_channel")
    ap.add_argument("--preproc-lt", type=Path, default=None)
    ap.add_argument("--preproc-u", type=Path, default=None)
    ap.add_argument("--n-bootstrap", type=int, default=100)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lbfgs-steps", type=int, default=30)
    ap.add_argument("--early-stop-patience", type=int, default=20)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--by-row", action="store_true")
    ap.add_argument("--beta-only", action="store_true")
    ap.add_argument("--beta-shared", action="store_true", help="Colapsa betas por alternativa en un beta generico.")
    ap.add_argument("--tabular-only", action="store_true")
    ap.add_argument("--progress-every", type=int, default=1, help="Imprime progreso cada N bootstraps.")
    ap.add_argument("--iter-log-every", type=int, default=0, help="Imprime log cada N iteraciones internas (0=off).")
    ap.add_argument("--utility-only", action="store_true", help="Entrena solo beta/delta/ASC para acelerar.")
    ap.add_argument("--skip-video", action="store_true", help="Salta encoder de video (usa ceros).")
    ap.add_argument("--skip-audio", action="store_true", help="Salta encoder de audio (usa ceros).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", type=Path, default=Path("results/MM_ICLV/bootstrap_param_summary.csv"))
    ap.add_argument("--split-info", type=Path, default=Path("results/MM_ICLV/split_info.txt"))
    args = ap.parse_args()

    run_hash = args.run_hash
    config = {}
    if run_hash is None:
        run_hash, config = parse_run_index(args.results_dir / "run_index.txt")

    pkl_path = args.pkl or Path(config.get("pkl", ""))
    if not pkl_path:
        raise ValueError("Se requiere --pkl o un run_index con pkl.")

    label_col = args.label_col or config.get("label_col", "action_proc")
    obs_lt_cols_file = args.obs_lt_cols_file or config.get("obs_lt_cols_file")
    obs_u_cols_file = args.obs_u_cols_file or config.get("obs_u_cols_file")
    ind_cols_file = args.indicator_cols_file or config.get("indicator_cols_file")

    preproc_lt_path = args.preproc_lt or (args.results_dir / f"MM_ICLV-preproc_lt-{run_hash}.pkl")
    preproc_u_path = args.preproc_u or (args.results_dir / f"MM_ICLV-preproc_u-{run_hash}.pkl")
    if not preproc_lt_path.exists() or not preproc_u_path.exists():
        raise FileNotFoundError("Faltan preprocesadores. Use --preproc-lt/--preproc-u o un run_hash valido.")

    df = pd.read_pickle(pkl_path) if pkl_path.suffix.lower() != ".csv" else pd.read_csv(pkl_path, low_memory=False)
    df = df.reset_index(drop=True)
    if label_col not in df.columns:
        raise ValueError(f"No existe label_col '{label_col}' en el dataframe.")

    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    if df[label_col].dtype == object:
        default_class_map = {
            "accelerate": 0,
            "brake": 1,
            "decelerate": 2,
            "maintain speed": 3,
            "wait": 4,
        }
        df[label_col] = df[label_col].map(default_class_map)
        df = df.dropna(subset=[label_col]).reset_index(drop=True)
    df[label_col] = df[label_col].astype(int)

    if args.path_col in df.columns:
        df = df[df[args.path_col].notna()].reset_index(drop=True)

    obs_lt_cols = load_features_file(obs_lt_cols_file) if obs_lt_cols_file else []
    obs_u_cols = load_features_file(obs_u_cols_file) if obs_u_cols_file else []
    ind_cols = load_features_file(ind_cols_file) if ind_cols_file else []
    obs_lt_cols = [c for c in obs_lt_cols if c in df.columns]
    obs_u_cols = [c for c in obs_u_cols if c in df.columns]
    ind_cols = [c for c in ind_cols if c in df.columns]

    participant_col = args.participant_col or config.get("participant_col", "participant")
    train_parts = set(parse_train_parts(args.split_info))
    if train_parts and participant_col in df.columns:
        df_train = df[df[participant_col].isin(train_parts)].copy()
    else:
        df_train = df

    preproc_lt = joblib.load(preproc_lt_path)
    preproc_u = joblib.load(preproc_u_path)

    # fallback a columnas del preproc si no se pudieron resolver desde archivos
    if not obs_lt_cols and hasattr(preproc_lt, "feature_names_in_"):
        obs_lt_cols = list(preproc_lt.feature_names_in_)
    if not obs_u_cols and hasattr(preproc_u, "feature_names_in_"):
        obs_u_cols = list(preproc_u.feature_names_in_)

    missing_lt = [c for c in obs_lt_cols if c not in df.columns]
    missing_u = [c for c in obs_u_cols if c not in df.columns]
    if missing_lt:
        raise ValueError(f"Faltan columnas OBS_LT en df: {missing_lt}")
    if missing_u:
        raise ValueError(f"Faltan columnas OBS_U en df: {missing_u}")

    X_lt_all = to_float_array(preproc_lt.transform(convertir_a_categorico(categorias_a_str(df[obs_lt_cols].copy()))))
    X_u_all = to_float_array(preproc_u.transform(convertir_a_categorico(categorias_a_str(df[obs_u_cols].copy()))))
    X_i_all = encode_indicator_blocks(df_train, df, ind_cols)

    y = df[label_col].to_numpy(dtype=np.int64)
    n_choices = int(pd.Series(y).nunique())

    base_model = MultimodalICLVDeterministic(
        tab_in_dim=X_lt_all.shape[1],
        dim_obs_u=X_u_all.shape[1],
        n_indicators=X_i_all.shape[1],
        n_choices=n_choices,
        tab_emb_dim=int(config.get("tab_emb_dim", 128)),
        shared_dim=int(config.get("n_latent", 64)),
        alpha=float(config.get("alpha", 1.0)),
        delta_per_alt=not bool(config.get("delta_shared", False)),
        fuse_dropout=0.0,
        freeze_video=bool(config.get("freeze_video", True)),
        freeze_audio=bool(config.get("freeze_audio", False)),
    )
    ckpt = args.results_dir / f"MM_ICLV-model-{run_hash}.pt"
    if ckpt.exists():
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location="cpu")
        base_model.load_state_dict(state, strict=False)
    base_model = base_model.to(torch.device(args.device))
    if args.skip_video:
        base_model.skip_video = True
    if args.skip_audio:
        base_model.skip_audio = True
    base_model.eval()

    print(f"[bootstrap] beta_per_alt={not args.beta_shared} obs_u_buy_only=False")

    if args.utility_only:
        for n, p in base_model.named_parameters():
            if not (n.startswith("beta") or n.startswith("delta") or n.startswith("ASC")):
                p.requires_grad = False
    base_params, base_names = expand_param_names(base_model)
    try:
        u_names = list(preproc_u.get_feature_names_out(obs_u_cols))
    except Exception:
        u_names = obs_u_cols

    total_params, trainable_params = _count_params(base_model)
    print(f"[bootstrap] params total={total_params} trainable={trainable_params}")

    params = []
    rng = np.random.default_rng(args.seed)
    grouped = df.groupby(participant_col).indices if participant_col in df.columns else {}
    uniq_parts = df[participant_col].dropna().unique() if participant_col in df.columns else []
    for b in range(args.n_bootstrap):
        t0 = time.time()
        if not args.by_row and len(uniq_parts) > 0:
            subs = rng.choice(uniq_parts, size=len(uniq_parts), replace=True)
            idx_list = [grouped[s] for s in subs if s in grouped]
            idx = np.concatenate(idx_list) if idx_list else bootstrap_indices(len(y), args.seed + b)
        else:
            idx = bootstrap_indices(len(y), args.seed + b)

        df_b = df.iloc[idx].reset_index(drop=True)
        X_lt_b = X_lt_all[idx]
        X_u_b = X_u_all[idx]
        X_i_b = X_i_all[idx]
        y_b = y[idx]

        base = MultimodalAudioDataset(
            df=df_b,
            tab_columns=obs_lt_cols,
            X_tab_array=torch.tensor(X_lt_b, dtype=torch.float32),
            path_col=args.path_col,
            label_col=label_col,
            timestamp_col=args.timestamp_col,
            window_id_col=args.window_id_col,
            participant_col=participant_col,
            audio_start_col=args.audio_start_col,
            audio_cached_col=args.audio_cached_col,
            audio_root=args.audio_root,
            audio_template=args.audio_template,
            audio_fallback_template=args.audio_fallback_template,
            audio_sr=args.audio_sr,
            audio_duration=args.audio_duration,
            audio_norm=args.audio_norm,
        )
        obs_u_t = torch.tensor(X_u_b, dtype=torch.float32)
        ind_t = torch.tensor(X_i_b, dtype=torch.float32)
        ds = MultimodalICLVDataset(base, obs_u_t, ind_t, n_choices=n_choices)
        loader = torch.utils.data.DataLoader(ds, batch_size=int(config.get("batch_size", 4)), shuffle=True, collate_fn=collate_multimodal_icl_v)

        model = MultimodalICLVDeterministic(
            tab_in_dim=X_lt_all.shape[1],
            dim_obs_u=X_u_all.shape[1],
            n_indicators=X_i_all.shape[1],
            n_choices=n_choices,
            tab_emb_dim=int(config.get("tab_emb_dim", 128)),
            shared_dim=int(config.get("n_latent", 64)),
            alpha=float(config.get("alpha", 1.0)),
            delta_per_alt=not bool(config.get("delta_shared", False)),
            fuse_dropout=0.0,
            freeze_video=bool(config.get("freeze_video", True)),
            freeze_audio=bool(config.get("freeze_audio", False)),
        ).to(torch.device(args.device))
        if ckpt.exists():
            try:
                state = torch.load(ckpt, map_location=args.device, weights_only=True)
            except TypeError:
                state = torch.load(ckpt, map_location=args.device)
            model.load_state_dict(state, strict=False)
        if args.skip_video:
            model.skip_video = True
        if args.skip_audio:
            model.skip_audio = True
        if args.utility_only:
            for n, p in model.named_parameters():
                if not (n.startswith("beta") or n.startswith("delta") or n.startswith("ASC")):
                    p.requires_grad = False

        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_ll = -np.inf
        patience = 0
        for _ in range(args.max_iter):
            ll = _epoch_pass(model, loader, torch.device(args.device), optimizer=opt)
            if args.iter_log_every and (_ + 1) % args.iter_log_every == 0:
                print(f"[bootstrap] iter={_+1}/{args.max_iter} ll={ll:.4f}")
            if ll > best_ll + 1e-6:
                best_ll = ll
                patience = 0
            else:
                patience += 1
                if patience >= args.early_stop_patience:
                    break

        if args.lbfgs_steps > 0:
            lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=args.lbfgs_steps, line_search_fn="strong_wolfe")

            def closure():
                lbfgs.zero_grad()
                ll_full = _epoch_pass(model, loader, torch.device(args.device), optimizer=None)
                loss = -torch.tensor(ll_full, device=torch.device(args.device), dtype=torch.float32)
                loss.backward()
                return loss

            lbfgs.step(closure)

        params_list, _ = expand_param_names(model)
        vec = torch.nn.utils.parameters_to_vector(params_list).detach().cpu().numpy()
        params.append(vec)
        if args.progress_every and (b + 1) % args.progress_every == 0:
            dt = time.time() - t0
            print(f"[bootstrap] done {b+1}/{args.n_bootstrap} (sec={dt:.2f})")

    samples = np.vstack(params)
    if args.beta_shared:
        beta_idx = [i for i, n in enumerate(base_names) if n.startswith("beta")]
        non_beta_idx = [i for i, n in enumerate(base_names) if not n.startswith("beta")]
        if not beta_idx:
            raise ValueError("No hay betas para colapsar (beta_shared).")
        dim_u = len(u_names)
        if dim_u <= 0:
            if n_choices > 0 and len(beta_idx) % n_choices == 0:
                dim_u = len(beta_idx) // n_choices
            else:
                raise ValueError("No se pudo inferir dim_obs_u para beta_shared.")
        beta_samples = samples[:, beta_idx].reshape(samples.shape[0], n_choices, dim_u)
        beta_shared = beta_samples.mean(axis=1)
        shared_names = [f"beta_shared[{i}]" for i in range(dim_u)]
        samples = np.concatenate([beta_shared, samples[:, non_beta_idx]], axis=1)
        base_names = shared_names + [base_names[i] for i in non_beta_idx]
    if args.beta_only and args.tabular_only:
        raise ValueError("--beta-only y --tabular-only son excluyentes.")
    if args.beta_only:
        keep_idx = [i for i, n in enumerate(base_names) if n.startswith("beta")]
        samples = samples[:, keep_idx]
        base_names = [base_names[i] for i in keep_idx]
    elif args.tabular_only:
        keep_idx = [i for i, n in enumerate(base_names) if n.startswith("beta") or n.startswith("tab_enc")]
        samples = samples[:, keep_idx]
        base_names = [base_names[i] for i in keep_idx]

    summary = summarize_params(samples, base_names)
    annotations = build_param_annotations(base_names, u_names, ind_cols, n_choices)
    summary = annotations.merge(summary, on="name", how="left")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv} (rows: {len(summary)})")


if __name__ == "__main__":
    main()
