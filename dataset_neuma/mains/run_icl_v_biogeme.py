from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, Draws, MonteCarlo, log, exp
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def load_cols(path: Path) -> list[str]:
    return [c.strip().lower() for c in path.read_text().splitlines() if c.strip()]


def warn_missing(df: pd.DataFrame, cols: list[str], label: str) -> list[str]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[warn] {label} missing cols: {missing}")
    return [c for c in cols if c in df.columns]


def stars_from_p(p: float) -> str:
    if p is None or np.isnan(p):
        return ""
    if p <= 0.01:
        return "***"
    if p <= 0.05:
        return "**"
    if p <= 0.1:
        return "*"
    return ""


def expand_categoricals(
    df: pd.DataFrame,
    cols: list[str],
    prefix: str,
    cat_unique_threshold: int,
    standardize_numeric: bool,
) -> tuple[pd.DataFrame, list[str]]:
    if not cols:
        return df, cols
    cat_cols = []
    num_cols = []
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique(dropna=True) <= cat_unique_threshold:
            cat_cols.append(c)
        else:
            num_cols.append(c)
    # Standardize numeric only (mean 0, std 1)
    if standardize_numeric and num_cols:
        for c in num_cols:
            col = pd.to_numeric(df[c], errors="coerce")
            mu = col.mean()
            sd = col.std()
            if sd and not np.isnan(sd):
                df[c] = (col - mu) / sd
            else:
                df[c] = col - mu
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
    parser.add_argument("--n-latent", type=int, default=1)
    parser.add_argument("--cat-unique-threshold", type=int, default=4)
    parser.add_argument("--standardize-numeric-only", action="store_true", help="Estandariza solo numéricas.")
    parser.add_argument("--minimal", action="store_true", help="Usa subconjuntos pequeños para facilitar convergencia.")
    parser.add_argument("--max-obs-u", type=int, default=5, help="Max columnas obs_u si --minimal.")
    parser.add_argument("--max-obs-i", type=int, default=1, help="Max columnas obs_i si --minimal.")
    parser.add_argument("--max-obs-lt", type=int, default=0, help="Max columnas obs_lt si --minimal (0 = todas).")
    parser.add_argument("--optimizer", type=str, default="BFGS", help="Algoritmo de optimizacion Biogeme.")
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
    if args.minimal:
        if args.max_obs_lt and args.max_obs_lt > 0:
            obs_lt_cols = obs_lt_cols[: args.max_obs_lt]
        obs_u_cols = obs_u_cols[: args.max_obs_u]
        obs_i_cols = obs_i_cols[: args.max_obs_i]
        print(f"[biogeme] minimal obs_lt={len(obs_lt_cols)} obs_u={len(obs_u_cols)} obs_i={len(obs_i_cols)}")

    # Convertir categóricas a dummies (Biogeme requiere float)
    df, obs_lt_cols = expand_categoricals(
        df, obs_lt_cols, prefix="lt_", cat_unique_threshold=args.cat_unique_threshold, standardize_numeric=args.standardize_numeric_only
    )
    df, obs_u_cols = expand_categoricals(
        df, obs_u_cols, prefix="u_", cat_unique_threshold=args.cat_unique_threshold, standardize_numeric=args.standardize_numeric_only
    )
    df, obs_i_cols = expand_categoricals(
        df, obs_i_cols, prefix="i_", cat_unique_threshold=args.cat_unique_threshold, standardize_numeric=args.standardize_numeric_only
    )

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

    # Latent variables: LV_k = Gamma_k * X + omega_k (Normal)
    gamma_betas = [
        [Beta(f"gamma_{k}_{c}", 0, None, None, 0) for c in obs_lt_cols]
        for k in range(args.n_latent)
    ]
    omegas = [Draws(f"omega_{k}", "NORMAL") for k in range(args.n_latent)]
    LVs = [
        sum(b * x for b, x in zip(gamma_betas[k], obs_lt_vars)) + omegas[k]
        for k in range(args.n_latent)
    ]

    # Utility (binary logit, base alt=0)
    ASC1 = Beta("ASC_1", 0, None, None, 0)
    beta_u = [Beta(f"beta_{c}", 0, None, None, 0) for c in obs_u_cols]
    delta = [Beta(f"delta_lv_{k}", 0, None, None, 0) for k in range(args.n_latent)]
    U1 = ASC1 + sum(b * x for b, x in zip(beta_u, obs_u_vars)) + sum(d * lv for d, lv in zip(delta, LVs))
    V = {0: 0, 1: U1}
    av = {0: 1, 1: 1}
    P = models.logit(V, av, Choice)

    # Measurement: indicators ~ Normal(alpha + lambda * LV, sigma)
    meas_loglik = 0
    for idx, y in enumerate(obs_i_vars):
        alpha = Beta(f"alpha_i{idx}", 0, None, None, 0)
        # Structured loadings for identification:
        # - If n_latent == 1: fix lambda for first indicator to 1
        # - If n_latent >= 2: fix indicator 0 to LV0, indicator 1 to LV1
        lam = []
        for k in range(args.n_latent):
            fixed = 0
            init = 0
            if args.n_latent == 1 and idx == 0 and k == 0:
                fixed = 1
                init = 1
            elif args.n_latent >= 2:
                if idx == 0 and k == 0:
                    fixed = 1
                    init = 1
                elif idx == 1 and k == 1:
                    fixed = 1
                    init = 1
            lam.append(Beta(f"lambda_i{idx}_lv{k}", init, None, None, fixed))

        # Fix sigma as constant = 1.0 (avoid estimation instability)
        sigma = 1.0
        mu = alpha + sum(lam_k * lv for lam_k, lv in zip(lam, LVs))
        z = (y - mu) / sigma
        log_pdf = -0.5 * (np.log(2 * np.pi) + 2 * log(sigma) + z * z)
        meas_loglik += log_pdf

    integrand = P * exp(meas_loglik)
    logprob = log(MonteCarlo(integrand))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    biogeme = bio.BIOGEME(database, logprob, number_of_draws=args.n_draws)
    biogeme.algorithm = args.optimizer
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

    if "stars" not in out.columns:
        out["stars"] = out["pval"].apply(stars_from_p)
    out_path = args.results_dir / "biogeme_params.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved params: {out_path}")
    print("General stats:", general)

    # Prediccion + metricas
    try:
        beta_values = results.getBetaValues()
        P1 = models.logit(V, av, 1)
        sim = bio.BIOGEME(database, {"p1": P1}, number_of_draws=args.n_draws)
        sim.model_name = "icl_v_biogeme_sim"
        sim_res = sim.simulate(beta_values)
        p1 = sim_res["p1"].to_numpy(dtype=float)
        y = df[label_col].to_numpy(dtype=int)
        p1 = np.clip(p1, 1e-9, 1 - 1e-9)
        y_hat = (p1 >= 0.5).astype(int)
        acc = float(accuracy_score(y, y_hat))
        f1_macro = float(f1_score(y, y_hat, average="macro"))
        f1_pos = float(f1_score(y, y_hat, pos_label=1))
        f1_neg = float(f1_score(y, y_hat, pos_label=0))
        auc = float(roc_auc_score(y, p1)) if len(np.unique(y)) > 1 else float("nan")
        loglik = float(np.sum(y * np.log(p1) + (1 - y) * np.log(1 - p1)))
        nll = float(-loglik)
        mean_nll = float(-loglik / max(1, len(y)))
        k = len(beta_values)
        aic = float(2 * k - 2 * loglik)
        bic = float(np.log(max(1, len(y))) * k - 2 * loglik)
        p_null = y.mean()
        p_null = float(min(max(p_null, 1e-9), 1 - 1e-9))
        loglik_null = float(np.sum(y * np.log(p_null) + (1 - y) * np.log(1 - p_null)))
        loglik_ratio = float(2 * (loglik - loglik_null))
        pseudo_r2 = float(1 - (loglik / loglik_null)) if loglik_null != 0 else float("nan")
        metrics = {
            "acc": acc,
            "f1_macro": f1_macro,
            "f1_pos": f1_pos,
            "f1_neg": f1_neg,
            "auc": auc,
            "nll": nll,
            "mean_nll": mean_nll,
            "log_likelihood": loglik,
            "aic": aic,
            "bic": bic,
            "loglik_null": loglik_null,
            "loglik_ratio": loglik_ratio,
            "pseudo_r2": pseudo_r2,
            "n_params": k,
            "n_obs": int(len(y)),
        }
        metrics_path = args.results_dir / "biogeme_metrics.json"
        pd.Series(metrics).to_json(metrics_path, indent=2, force_ascii=False)
        print("Saved metrics:", metrics_path)
    except Exception as exc:
        print(f"[warn] no se pudieron calcular metricas predictivas: {exc}")


if __name__ == "__main__":
    main()
