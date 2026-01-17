from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, Draws, MonteCarlo, log, exp


def load_cols(path: Path) -> list[str]:
    return [c.strip().lower() for c in path.read_text().splitlines() if c.strip()]


def warn_missing(df: pd.DataFrame, cols: list[str], label: str) -> list[str]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[warn] {label} missing cols: {missing}")
    return [c for c in cols if c in df.columns]


def expand_categoricals(df: pd.DataFrame, cols: list[str], prefix: str) -> tuple[pd.DataFrame, list[str]]:
    if not cols:
        return df, cols
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in cols if c not in cat_cols]
    if not cat_cols:
        return df, cols
    dummies = pd.get_dummies(df[cat_cols].astype(str), prefix=[f"{prefix}{c}" for c in cat_cols], drop_first=True)
    df = df.drop(columns=cat_cols).join(dummies)
    new_cols = num_cols + dummies.columns.tolist()
    print(f"[biogeme] one-hot cols: {cat_cols} -> {len(dummies.columns)} dummies")
    return df, new_cols


def main() -> None:
    parser = argparse.ArgumentParser(description="ICLV minimo en Biogeme.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--obs-lt-cols", type=Path, required=True)
    parser.add_argument("--obs-u-cols", type=Path, required=True)
    parser.add_argument("--obs-i-cols", type=Path, required=True)
    parser.add_argument("--label-col", type=str, default="bought")
    parser.add_argument("--n-draws", type=int, default=200)
    parser.add_argument("--results-dir", type=Path, default=Path("./results/icl_v_biogeme"))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.lower()
    label_col = args.label_col.lower()
    if label_col not in df.columns:
        raise ValueError(f"No se encontro etiqueta '{label_col}'.")

    obs_lt_cols = warn_missing(df, load_cols(args.obs_lt_cols), "obs_lt")
    obs_u_cols = warn_missing(df, load_cols(args.obs_u_cols), "obs_u")
    obs_i_cols = warn_missing(df, load_cols(args.obs_i_cols), "obs_i")

    # Convertir categóricas a dummies (Biogeme requiere float)
    df, obs_lt_cols = expand_categoricals(df, obs_lt_cols, prefix="lt_")
    df, obs_u_cols = expand_categoricals(df, obs_u_cols, prefix="u_")
    df, obs_i_cols = expand_categoricals(df, obs_i_cols, prefix="i_")

    # Coerce numeric + drop NaN en columnas relevantes
    keep_cols = [label_col] + obs_lt_cols + obs_u_cols + obs_i_cols
    df = df[keep_cols].copy()
    for c in keep_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    database = db.Database("neuma", df)

    # Variables
    Choice = Variable(label_col)
    obs_lt_vars = [Variable(c) for c in obs_lt_cols]
    obs_u_vars = [Variable(c) for c in obs_u_cols]
    obs_i_vars = [Variable(c) for c in obs_i_cols]

    # Latent variable: LV = Gamma * X + omega (Normal)
    gamma_betas = [Beta(f"gamma_{c}", 0, None, None, 0) for c in obs_lt_cols]
    omega = Draws("omega", "NORMAL")
    LV = sum(b * x for b, x in zip(gamma_betas, obs_lt_vars)) + omega

    # Utility (binary logit, base alt=0)
    ASC1 = Beta("ASC_1", 0, None, None, 0)
    beta_u = [Beta(f"beta_{c}", 0, None, None, 0) for c in obs_u_cols]
    delta = Beta("delta_lv", 0, None, None, 0)
    U1 = ASC1 + sum(b * x for b, x in zip(beta_u, obs_u_vars)) + delta * LV
    V = {0: 0, 1: U1}
    av = {0: 1, 1: 1}
    P = models.logit(V, av, Choice)

    # Measurement: indicators ~ Normal(alpha + lambda * LV, sigma)
    meas_loglik = 0
    for idx, y in enumerate(obs_i_vars):
        alpha = Beta(f"alpha_i{idx}", 0, None, None, 0)
        lam = Beta(f"lambda_i{idx}", 1, None, None, 0)
        sigma = Beta(f"sigma_i{idx}", 1, 1e-6, None, 0)
        mu = alpha + lam * LV
        z = (y - mu) / sigma
        log_pdf = -0.5 * (np.log(2 * np.pi) + 2 * log(sigma) + z * z)
        meas_loglik += log_pdf

    integrand = P * exp(meas_loglik)
    logprob = log(MonteCarlo(integrand))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    biogeme = bio.BIOGEME(database, logprob, number_of_draws=args.n_draws)
    biogeme.model_name = "icl_v_biogeme"
    results = biogeme.estimate()

    def get_with_fallback(obj, names):
        for name in names:
            if hasattr(obj, name):
                try:
                    return getattr(obj, name)()
                except TypeError:
                    return getattr(obj, name)
        raw = getattr(obj, "raw_estimation_results", None)
        if raw is not None:
            for name in names:
                if hasattr(raw, name):
                    try:
                        return getattr(raw, name)()
                    except TypeError:
                        return getattr(raw, name)
        return None

    betas = get_with_fallback(results, ["get_beta_values", "getBetaValues"])
    std = get_with_fallback(results, ["get_std_errors", "getStdErrValues", "getStdErr"])
    tstat = get_with_fallback(results, ["get_t_stats", "getTTest"])
    pval = get_with_fallback(results, ["get_p_values", "getPValues"])
    general = get_with_fallback(results, ["get_general_statistics", "getGeneralStatistics"])
    est_params = get_with_fallback(
        results,
        [
            "get_pandas_estimated_parameters",
            "get_estimated_parameters",
            "getEstimatedParameters",
        ],
    )

    out_rows = []
    if betas is None:
        raise RuntimeError("No se pudieron leer betas desde Biogeme.")
    for name in betas.keys():
        std_val = std[name] if std is not None and name in std else np.nan
        t_val = tstat[name] if tstat is not None and name in tstat else np.nan
        p_val = pval[name] if pval is not None and name in pval else np.nan
        out_rows.append({"name": name, "beta": betas[name], "std": std_val, "tstat": t_val, "pval": p_val})
    out = pd.DataFrame(out_rows)
    # Fallback from estimated parameters table when std/tstat are missing
    if est_params is not None:
        try:
            est = est_params if isinstance(est_params, pd.DataFrame) else pd.DataFrame(est_params)
            est.columns = [c.lower().strip().replace(" ", "_") for c in est.columns]
            if "parameter" not in est.columns and "name" in est.columns:
                est = est.rename(columns={"name": "parameter"})
            if "value" in est.columns and "parameter" in est.columns:
                est = est.set_index("parameter")
                col_candidates = {
                    "std": ["std_err", "robust_std_err.", "robust_std_err", "std_err."],
                    "tstat": ["t_test", "robust_t-stat.", "robust_t_stat", "t_stat", "t_test."],
                    "pval": ["p_value", "robust_p-value", "robust_p_value", "p_value."],
                }
                for dst, candidates in col_candidates.items():
                    for src in candidates:
                        if src in est.columns:
                            out[dst] = out[dst].fillna(out["name"].map(est[src]))
                            break
            else:
                print("[warn] estimated_parameters columns:", est.columns.tolist())
        except Exception as exc:
            print(f"[warn] failed to parse estimated_parameters: {exc}")

    out_path = args.results_dir / "biogeme_params.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved params: {out_path}")
    print("General stats:", general)


if __name__ == "__main__":
    main()
