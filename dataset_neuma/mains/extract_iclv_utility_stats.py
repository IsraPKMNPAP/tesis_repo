from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch


def load_feature_names_from_preproc(preproc_path: Path, obs_u_cols: List[str]) -> List[str]:
    try:
        preproc = torch.load(preproc_path)
        if hasattr(preproc, "get_feature_names_out"):
            return list(preproc.get_feature_names_out(obs_u_cols))
    except Exception:
        pass
    return obs_u_cols


def extract_beta_from_state(state: dict) -> Tuple[np.ndarray, int]:
    if "beta.weight" in state:
        beta = state["beta.weight"].detach().cpu().numpy()
        if beta.ndim == 2 and beta.shape[0] == 1:
            beta = beta[0:1, :]
        return beta, beta.shape[0]
    if "beta" in state:
        beta = state["beta"].detach().cpu().numpy()
        if beta.ndim == 1:
            beta = beta.reshape(1, -1)
        return beta, beta.shape[0]
    raise ValueError("No se encontro beta en state_dict.")


def load_hessian_stats(hessian_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    data = json.loads(hessian_path.read_text(encoding="utf-8"))
    theta = np.asarray(data["theta"])
    std = np.asarray(data["std"])
    names = list(data["names"])
    return theta, std, names


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae betas y estadisticos de utilidad ICLV.")
    parser.add_argument("--iclv-dir", type=Path, default=Path("./results/icl_v_standard/run_0001"))
    parser.add_argument("--mm-dir", type=Path, default=Path("./results/multimodal_icl_v_standard/run_0001"))
    args = parser.parse_args()

    rows = []

    # ICLV clasico
    iclv_metrics = json.loads((args.iclv_dir / "metrics.json").read_text(encoding="utf-8"))
    obs_u_cols = iclv_metrics.get("obs_u_cols", [])
    feature_names = load_feature_names_from_preproc(args.iclv_dir / "preproc_u.pkl", obs_u_cols)
    state = torch.load(args.iclv_dir / "model.pt", map_location="cpu")
    beta, n_alts = extract_beta_from_state(state)

    if (args.iclv_dir / "hessian.json").exists():
        theta, std, names = load_hessian_stats(args.iclv_dir / "hessian.json")
        beta_idx = [i for i, n in enumerate(names) if n.startswith("beta")]
        for alt in range(n_alts):
            for j, feat in enumerate(feature_names):
                idx = alt * beta.shape[1] + j
                if idx >= beta.shape[1] * n_alts:
                    continue
                flat_idx = beta_idx[idx] if idx < len(beta_idx) else None
                coef = beta[alt, j]
                sd = std[flat_idx] if (flat_idx is not None and flat_idx < len(std)) else np.nan
                tstat = coef / sd if sd == sd and sd != 0 else np.nan
                rows.append(
                    {
                        "model": "icl_v",
                        "alt": alt,
                        "feature": feat,
                        "coef": float(coef),
                        "std": float(sd) if sd == sd else np.nan,
                        "tstat": float(tstat) if tstat == tstat else np.nan,
                    }
                )
    else:
        for alt in range(n_alts):
            for j, feat in enumerate(feature_names):
                rows.append(
                    {
                        "model": "icl_v",
                        "alt": alt,
                        "feature": feat,
                        "coef": float(beta[alt, j]),
                        "std": np.nan,
                        "tstat": np.nan,
                    }
                )

    # ICLV multimodal
    mm_metrics = json.loads((args.mm_dir / "metrics.json").read_text(encoding="utf-8"))
    mm_obs_u_cols = mm_metrics.get("obs_u_cols", [])
    state_mm = torch.load(args.mm_dir / "model_last.pt", map_location="cpu")
    beta_mm, n_alts_mm = extract_beta_from_state(state_mm)
    if (args.mm_dir / "hessian.json").exists():
        theta, std, names = load_hessian_stats(args.mm_dir / "hessian.json")
        beta_idx = [i for i, n in enumerate(names) if n.startswith("beta")]
        for alt in range(n_alts_mm):
            for j, feat in enumerate(mm_obs_u_cols):
                idx = alt * beta_mm.shape[1] + j
                flat_idx = beta_idx[idx] if idx < len(beta_idx) else None
                coef = beta_mm[alt, j]
                sd = std[flat_idx] if (flat_idx is not None and flat_idx < len(std)) else np.nan
                tstat = coef / sd if sd == sd and sd != 0 else np.nan
                rows.append(
                    {
                        "model": "multimodal_icl_v",
                        "alt": alt,
                        "feature": feat,
                        "coef": float(coef),
                        "std": float(sd) if sd == sd else np.nan,
                        "tstat": float(tstat) if tstat == tstat else np.nan,
                    }
                )
    else:
        for alt in range(n_alts_mm):
            for j, feat in enumerate(mm_obs_u_cols):
                rows.append(
                    {
                        "model": "multimodal_icl_v",
                        "alt": alt,
                        "feature": feat,
                        "coef": float(beta_mm[alt, j]),
                        "std": np.nan,
                        "tstat": np.nan,
                    }
                )

    out_df = pd.DataFrame(rows)
    iclv_out = args.iclv_dir / "utility_stats.csv"
    mm_out = args.mm_dir / "utility_stats.csv"
    out_df[out_df["model"] == "icl_v"].to_csv(iclv_out, index=False)
    out_df[out_df["model"] == "multimodal_icl_v"].to_csv(mm_out, index=False)
    print(f"Saved: {iclv_out} (rows: {len(out_df[out_df['model']=='icl_v'])})")
    print(f"Saved: {mm_out} (rows: {len(out_df[out_df['model']=='multimodal_icl_v'])})")


if __name__ == "__main__":
    main()
