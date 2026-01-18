from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn.utils import parameters_to_vector

# Ensure package root on path (dataset_bicicletas/)
ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.icl_v import DeterministicICLV
from src.data_cleaning.cleaning import categorias_a_str, convertir_a_categorico
from utils.features import load_features_file


def parse_run_index(idx_path: Path) -> Tuple[str, dict]:
    if not idx_path.exists():
        raise FileNotFoundError(f"No existe {idx_path}")
    blocks = [b.strip() for b in idx_path.read_text(encoding="utf-8").split("-----") if b.strip()]
    for block in reversed(blocks):
        if "model=ICLV" not in block:
            continue
        m_hash = re.search(r"hash=(\w+)", block)
        m_cfg = re.search(r"config:\s*(\{.*\})", block, flags=re.S)
        if not m_hash or not m_cfg:
            continue
        run_hash = m_hash.group(1)
        config = json.loads(m_cfg.group(1))
        return run_hash, config
    raise ValueError("No se encontro un run ICLV en run_index.txt")


def parse_train_parts(split_path: Path) -> List[str]:
    if not split_path.exists():
        return []
    text = split_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("Train:"):
            if "->" in line:
                tail = line.split("->", 1)[1].strip()
                try:
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


def stars_for_t(t: float) -> str:
    if np.isnan(t):
        return ""
    if abs(t) >= 2.58:
        return "***"
    if abs(t) >= 1.96:
        return "**"
    if abs(t) >= 1.64:
        return "*"
    return ""


def build_param_metadata(
    model: DeterministicICLV,
    feat_names_u: Sequence[str],
    feat_names_lt: Sequence[str],
    indicator_cols: Sequence[str],
    n_choices: int,
) -> tuple[List[torch.nn.Parameter], List[dict]]:
    params: List[torch.nn.Parameter] = []
    meta: List[dict] = []
    base_alt = getattr(model, "base_alt", 0)
    alt_list = [a for a in range(n_choices) if a != base_alt]
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        params.append(p)
        if name == "beta" and model.beta_per_alt:
            for alt_pos, alt in enumerate(alt_list):
                for j, feat in enumerate(feat_names_u):
                    meta.append(
                        {
                            "name": f"{name}[{alt_pos},{j}]",
                            "block": "utility",
                            "var_name": feat,
                            "alt": alt,
                            "latent": None,
                        }
                    )
        elif name == "beta.weight" and not model.beta_per_alt:
            for j, feat in enumerate(feat_names_u):
                meta.append(
                    {
                        "name": f"{name}[0,{j}]",
                        "block": "utility",
                        "var_name": feat,
                        "alt": "all",
                        "latent": None,
                    }
                )
        elif name == "delta":
            if p.dim() == 2:
                for alt_pos, alt in enumerate(alt_list):
                    for lv in range(p.shape[1]):
                        meta.append(
                            {
                                "name": f"{name}[{alt_pos},{lv}]",
                                "block": "utility",
                                "var_name": f"LV{lv}",
                                "alt": alt,
                                "latent": lv,
                            }
                        )
            else:
                for lv in range(p.numel()):
                    meta.append(
                        {
                            "name": f"{name}[{lv}]",
                            "block": "utility",
                            "var_name": f"LV{lv}",
                            "alt": "all",
                            "latent": lv,
                        }
                    )
        elif name == "ASC":
            for alt_pos, alt in enumerate(alt_list):
                meta.append(
                    {
                        "name": f"{name}[{alt_pos}]",
                        "block": "utility",
                        "var_name": "ASC",
                        "alt": alt,
                        "latent": None,
                    }
                )
        elif name == "Gamma.weight":
            for lv in range(p.shape[0]):
                for j, feat in enumerate(feat_names_lt):
                    meta.append(
                        {
                            "name": f"{name}[{lv},{j}]",
                            "block": "structural",
                            "var_name": feat,
                            "alt": None,
                            "latent": lv,
                        }
                    )
        elif name == "Gamma.bias":
            for lv in range(p.numel()):
                meta.append(
                    {
                        "name": f"{name}[{lv}]",
                        "block": "structural",
                        "var_name": "intercept",
                        "alt": None,
                        "latent": lv,
                    }
                )
        elif name == "Lambda.weight":
            for i, ind_name in enumerate(indicator_cols):
                for lv in range(p.shape[1]):
                    meta.append(
                        {
                            "name": f"{name}[{i},{lv}]",
                            "block": "measurement",
                            "var_name": ind_name,
                            "alt": None,
                            "latent": lv,
                        }
                    )
        elif name == "Lambda.bias":
            for i, ind_name in enumerate(indicator_cols):
                meta.append(
                    {
                        "name": f"{name}[{i}]",
                        "block": "measurement",
                        "var_name": ind_name,
                        "alt": None,
                        "latent": None,
                    }
                )
        else:
            for idx in range(p.numel()):
                meta.append(
                    {
                        "name": f"{name}[{idx}]" if p.numel() > 1 else name,
                        "block": "other",
                        "var_name": name,
                        "alt": None,
                        "latent": None,
                    }
                )
    return params, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap completo para ICLV (dataset_bicicletas).")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--run-hash", type=str, default=None, help="Hash de run ICLV (si no, se usa el ultimo).")
    ap.add_argument("--pkl", type=Path, default=None)
    ap.add_argument("--label-col", type=str, default=None)
    ap.add_argument("--obs-lt-cols-file", type=str, default=None)
    ap.add_argument("--obs-u-cols-file", type=str, default=None)
    ap.add_argument("--indicator-cols-file", type=str, default=None)
    ap.add_argument("--preproc-lt", type=Path, default=None)
    ap.add_argument("--preproc-u", type=Path, default=None)
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--by-row", action="store_true", help="Bootstrap por filas (default: por participante).")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--beta-per-alt", action="store_true")
    ap.add_argument("--out-csv", type=Path, default=Path("results/ICLV/bootstrap_all_params.csv"))
    ap.add_argument("--split-info", type=Path, default=Path("results/ICLV/split_info.txt"))
    args = ap.parse_args()

    run_hash = args.run_hash
    config = {}
    if run_hash is None:
        run_hash, config = parse_run_index(args.results_dir / "run_index.txt")

    pkl_path = args.pkl or Path(config.get("pkl", ""))
    if not pkl_path:
        raise ValueError("Se requiere --pkl o un run_index con pkl.")

    label_col = args.label_col or config.get("label_col", "action_proc")
    obs_lt_cols_file = args.obs_lt_cols_file or config.get("obs_lt_cols_file") or "utils/feature_sets/filtered_iclv/obs_lt.txt"
    obs_u_cols_file = args.obs_u_cols_file or config.get("obs_u_cols_file") or "utils/feature_sets/filtered_iclv/obs_u.txt"
    ind_cols_file = args.indicator_cols_file or config.get("indicator_cols_file") or "utils/feature_sets/filtered_iclv/obs_i.txt"

    preproc_lt_path = args.preproc_lt or (args.results_dir / f"ICLV-preproc_lt-{run_hash}.pkl")
    preproc_u_path = args.preproc_u or (args.results_dir / f"ICLV-preproc_u-{run_hash}.pkl")
    if not preproc_lt_path.exists() or not preproc_u_path.exists():
        raise FileNotFoundError("Faltan preprocesadores. Use --preproc-lt/--preproc-u o un run_hash valido.")

    df = pd.read_pickle(pkl_path) if pkl_path.suffix.lower() != ".csv" else pd.read_csv(pkl_path, low_memory=False)
    df = df.reset_index(drop=True)

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

    obs_lt_cols = load_features_file(obs_lt_cols_file) if obs_lt_cols_file else []
    obs_u_cols = load_features_file(obs_u_cols_file) if obs_u_cols_file else []
    ind_cols = load_features_file(ind_cols_file) if ind_cols_file else []
    obs_lt_cols = [c for c in obs_lt_cols if c in df.columns]
    obs_u_cols = [c for c in obs_u_cols if c in df.columns]
    ind_cols = [c for c in ind_cols if c in df.columns]

    train_parts = set(parse_train_parts(args.split_info))
    if train_parts:
        df_train = df[df[config.get("participant_col", "participant")].isin(train_parts)].copy()
    else:
        df_train = df

    preproc_lt = joblib.load(preproc_lt_path)
    preproc_u = joblib.load(preproc_u_path)

    X_lt_all = to_float_array(preproc_lt.transform(convertir_a_categorico(categorias_a_str(df[obs_lt_cols].copy()))))
    X_u_all = to_float_array(preproc_u.transform(convertir_a_categorico(categorias_a_str(df[obs_u_cols].copy()))))
    X_i_all = encode_indicator_blocks(df_train, df, ind_cols)

    y = df[label_col].to_numpy(dtype=np.int64)
    n_choices = int(pd.Series(y).nunique())

    # Expand obs_u to [N, J, D]
    X_u = X_u_all[:, None, :].repeat(n_choices, axis=1)

    device = torch.device(args.device)
    n_latent = int(config.get("n_latent", 3))
    delta_per_alt = not bool(config.get("delta_shared", False))
    lr = args.lr if args.lr is not None else float(config.get("lr", 1e-3))
    alpha = float(config.get("alpha", 1.0))

    model_init = DeterministicICLV(
        dim_obs_lt=X_lt_all.shape[1],
        dim_obs_u=X_u.shape[2],
        n_latent=n_latent,
        n_indicators=X_i_all.shape[1],
        n_choices=n_choices,
        alpha=alpha,
        delta_per_alt=delta_per_alt,
        beta_per_alt=args.beta_per_alt,
    ).to(device)

    try:
        feat_names_u = list(preproc_u.get_feature_names_out(obs_u_cols))
    except Exception:
        feat_names_u = obs_u_cols
    try:
        feat_names_lt = list(preproc_lt.get_feature_names_out(obs_lt_cols))
    except Exception:
        feat_names_lt = obs_lt_cols

    _, param_meta = build_param_metadata(
        model_init,
        feat_names_u,
        feat_names_lt,
        ind_cols,
        n_choices,
    )

    param_samples = []
    participant_col = config.get("participant_col", "participant")
    by_participant = not args.by_row
    for b in range(args.n_bootstrap):
        if by_participant and participant_col in df.columns:
            rng = np.random.default_rng(args.seed + b)
            uniq = df[participant_col].dropna().unique()
            subs = rng.choice(uniq, size=len(uniq), replace=True)
            grouped = df.groupby(participant_col).indices
            idx_list = [grouped[s] for s in subs if s in grouped]
            idx = np.concatenate(idx_list) if idx_list else bootstrap_indices(len(y), args.seed + b)
        else:
            idx = bootstrap_indices(len(y), args.seed + b)

        obs_lt_b = torch.tensor(X_lt_all[idx], dtype=torch.float32, device=device)
        obs_u_b = torch.tensor(X_u[idx], dtype=torch.float32, device=device)
        ind_b = torch.tensor(X_i_all[idx], dtype=torch.float32, device=device)
        y_b = torch.tensor(y[idx], dtype=torch.long, device=device)

        model = DeterministicICLV(
            dim_obs_lt=X_lt_all.shape[1],
            dim_obs_u=X_u.shape[2],
            n_latent=n_latent,
            n_indicators=X_i_all.shape[1],
            n_choices=n_choices,
            alpha=alpha,
            delta_per_alt=delta_per_alt,
            beta_per_alt=args.beta_per_alt,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for _ in range(args.max_iter):
            out = model(obs_lt_b, obs_u_b, ind_b, y_b)
            loss = out["loss"]
            if args.l2 > 0:
                l2_term = sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)
                loss = loss + args.l2 * l2_term
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        param_vec = parameters_to_vector([p for p in model.parameters() if p.requires_grad]).detach().cpu().numpy()
        param_samples.append(param_vec)

    params = np.vstack(param_samples)
    mean = params.mean(axis=0)
    std = params.std(axis=0, ddof=1)
    tstat = mean / np.where(std == 0, np.nan, std)

    rows = []
    for meta, m, s, t in zip(param_meta, mean, std, tstat):
        rows.append(
            {
                "name": meta["name"],
                "block": meta["block"],
                "var_name": meta["var_name"],
                "alt": meta["alt"],
                "latent": meta["latent"],
                "mean": float(m),
                "std": float(s),
                "tstat": float(t),
                "stars": stars_for_t(t),
            }
        )
    out_df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved: {args.out_csv} (rows: {len(out_df)})")


if __name__ == "__main__":
    main()
