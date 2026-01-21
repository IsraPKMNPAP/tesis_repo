from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, MonteCarlo, log, exp
try:
    from biogeme.expressions import Draws
except Exception:
    # Newer biogeme exposes draw expressions as bioDraws
    from biogeme.expressions import bioDraws as Draws
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

pd.set_option("future.no_silent_downcasting", True)

# Ensure package root on path when running from dataset_bicicletas
ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.splits import split_by_participant


def load_cols(path: Path) -> list[str]:
    return [c.strip().lower() for c in path.read_text(encoding="utf-8").splitlines() if c.strip()]


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
        nunique = df[c].nunique(dropna=True)
        is_num = pd.api.types.is_numeric_dtype(df[c])
        if (not is_num) and nunique > cat_unique_threshold:
            print(f"[biogeme] dropped high-card categorical: {c} (unique={nunique})")
            continue
        if nunique <= cat_unique_threshold:
            cat_cols.append(c)
        elif is_num:
            num_cols.append(c)
        else:
            cat_cols.append(c)
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
    # Impute numeric (mean) and categorical (mode) before one-hot
    for c in num_cols:
        col = pd.to_numeric(df[c], errors="coerce")
        mu = col.mean()
        df[c] = col.fillna(mu)
    for c in cat_cols:
        mode = df[c].mode(dropna=True)
        fill_val = mode.iloc[0] if len(mode) else "missing"
        df[c] = df[c].fillna(fill_val).infer_objects(copy=False)
    if not cat_cols:
        return df, cols
    dummies = pd.get_dummies(df[cat_cols].astype(str), prefix=[f"{prefix}{c}" for c in cat_cols], drop_first=True)
    # Ensure dummy columns are numeric (biogeme rejects bool)
    dummies = dummies.astype(int)
    df = df.drop(columns=cat_cols).join(dummies)
    new_cols = num_cols + dummies.columns.tolist()
    print(f"[biogeme] one-hot cols: {cat_cols} -> {len(dummies.columns)} dummies")
    return df, new_cols


def sanitize_columns(df: pd.DataFrame, cols: list[str], label: str) -> tuple[pd.DataFrame, list[str]]:
    """Make column names safe for Biogeme variables and Betas."""
    import re

    def _safe(name: str) -> str:
        s = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_")
        if not s:
            s = "col"
        if s[0].isdigit():
            s = f"c_{s}"
        return s

    mapping = {}
    used = set(df.columns)
    for c in cols:
        safe = _safe(c)
        base = safe
        k = 1
        while safe in used and safe != c:
            safe = f"{base}_{k}"
            k += 1
        mapping[c] = safe
        used.add(safe)
    if mapping:
        df = df.rename(columns=mapping)
    renamed = [mapping.get(c, c) for c in cols]
    if mapping:
        print(f"[biogeme] sanitized {label} names (sample): {list(mapping.items())[:5]}")
    return df, renamed


def main() -> None:
    parser = argparse.ArgumentParser(description="ICLV en Biogeme (dataset_bicicletas).")
    parser.add_argument("--data", type=Path, required=True, help="CSV o PKL con dataset.")
    parser.add_argument("--obs-lt-cols", type=Path, required=True)
    parser.add_argument("--obs-u-cols", type=Path, required=True)
    parser.add_argument("--obs-i-cols", type=Path, required=True)
    parser.add_argument("--label-col", type=str, default="action_proc")
    parser.add_argument("--participant-col", type=str, default="participant")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--half-data", action="store_true", help="Usar 50% de participantes para acelerar")
    parser.add_argument("--data-frac", type=float, default=None, help="Fraccion de participantes a usar (0-1).")
    parser.add_argument("--n-draws", type=int, default=200)
    parser.add_argument("--n-latent", type=int, default=1)
    parser.add_argument("--mnl-only", action="store_true", help="Usa MNL sin latentes ni medicion.")
    parser.add_argument("--no-measurement", action="store_true", help="Usa latentes sin bloque de medicion.")
    parser.add_argument("--cat-unique-threshold", type=int, default=5)
    parser.add_argument("--standardize-numeric-only", action="store_true", default=True)
    parser.add_argument("--minimal", action="store_true")
    parser.add_argument("--limit-blocks", action="store_true", help="Limita cantidad de variables por bloque")
    parser.add_argument("--max-obs-u", type=int, default=5)
    parser.add_argument("--max-obs-i", type=int, default=1)
    parser.add_argument("--max-obs-lt", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="BFGS")
    parser.add_argument("--results-dir", type=Path, default=Path("results/icl_v_biogeme"))
    parser.add_argument("--model-name", type=str, default="icl_v_biogeme")
    args = parser.parse_args()
    if not args.model_name:
        args.model_name = "icl_v_biogeme"

    data_path = args.data
    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path, low_memory=False)
    else:
        df = pd.read_pickle(data_path)
    df.columns = df.columns.str.lower()
    label_col = args.label_col.lower()
    if label_col not in df.columns:
        raise ValueError(f"No se encontro etiqueta '{label_col}'.")

    obs_lt_cols = warn_missing(df, load_cols(args.obs_lt_cols), "obs_lt")
    obs_u_cols = warn_missing(df, load_cols(args.obs_u_cols), "obs_u")
    obs_i_cols = warn_missing(df, load_cols(args.obs_i_cols), "obs_i")
    if args.minimal or args.limit_blocks:
        if args.max_obs_lt and args.max_obs_lt > 0:
            obs_lt_cols = obs_lt_cols[: args.max_obs_lt]
        if args.max_obs_u and args.max_obs_u > 0:
            obs_u_cols = obs_u_cols[: args.max_obs_u]
        if args.limit_blocks and args.max_obs_i and args.max_obs_i > 0:
            obs_i_cols = obs_i_cols[: args.max_obs_i]
        if args.minimal:
            print(f"[biogeme] minimal obs_lt={len(obs_lt_cols)} obs_u={len(obs_u_cols)} obs_i={len(obs_i_cols)}")
        else:
            print(f"[biogeme] limited obs_lt={len(obs_lt_cols)} obs_u={len(obs_u_cols)} obs_i={len(obs_i_cols)}")

    # Map action labels if needed
    if df[label_col].dtype == object:
        label_map = {
            "accelerate": 0,
            "brake": 1,
            "decelerate": 2,
            "maintain speed": 3,
            "wait": 4,
        }
        df[label_col] = df[label_col].map(label_map)

    # Convert categoricals to dummies (Biogeme requires float)
    df, obs_lt_cols = expand_categoricals(
        df, obs_lt_cols, prefix="lt_", cat_unique_threshold=args.cat_unique_threshold, standardize_numeric=args.standardize_numeric_only
    )
    df, obs_u_cols = expand_categoricals(
        df, obs_u_cols, prefix="u_", cat_unique_threshold=args.cat_unique_threshold, standardize_numeric=args.standardize_numeric_only
    )
    df, obs_i_cols = expand_categoricals(
        df, obs_i_cols, prefix="i_", cat_unique_threshold=args.cat_unique_threshold, standardize_numeric=args.standardize_numeric_only
    )
    # Sanitize names for Biogeme
    df, obs_lt_cols = sanitize_columns(df, obs_lt_cols, "obs_lt")
    df, obs_u_cols = sanitize_columns(df, obs_u_cols, "obs_u")
    df, obs_i_cols = sanitize_columns(df, obs_i_cols, "obs_i")

    # Drop specific noisy column if present
    drop_u_col = "u_obstructions_obst_culos"
    if drop_u_col in df.columns:
        df = df.drop(columns=[drop_u_col])
        if drop_u_col in obs_u_cols:
            obs_u_cols = [c for c in obs_u_cols if c != drop_u_col]
        print(f"[biogeme] dropped noisy column: {drop_u_col}")

    # Coerce numeric + drop NaN in relevant columns
    keep_cols = [label_col, args.participant_col] + obs_lt_cols + obs_u_cols + obs_i_cols
    df = df[keep_cols].copy()
    # Coerce numeric columns where possible; keep categoricals for later imputation/one-hot
    for c in keep_cols:
        if c == label_col:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    if args.mnl_only:
        print("[biogeme] mnl_only=True: sin latentes ni bloque de medicion")
    if args.no_measurement and not args.mnl_only:
        print("[biogeme] no_measurement=True: con latentes, sin bloque de medicion")
    # Optional: reduce to half of participants before split
    frac = args.data_frac
    if args.half_data:
        frac = 0.5
    if frac is not None and args.participant_col in df.columns:
        parts = pd.Index(df[args.participant_col].dropna().unique())
        if len(parts) > 0:
            rng = np.random.default_rng(args.seed)
            k = max(1, int(np.ceil(len(parts) * float(frac))))
            keep_parts = rng.choice(parts, size=k, replace=False)
            df = df[df[args.participant_col].isin(keep_parts)].reset_index(drop=True)
            print(f"[biogeme] subset participants: {len(keep_parts)}/{len(parts)} (frac={float(frac)})")

    # Split by participant before estimation
    if args.participant_col in df.columns:
        df_tr, df_val, df_te, split_info = split_by_participant(
            df,
            participant_col=args.participant_col,
            val_frac=args.val_split,
            test_frac=args.test_split,
            seed=args.seed,
        )
    else:
        df_tr, df_val, df_te = df, df.iloc[0:0].copy(), df.iloc[0:0].copy()

    # Ensure labels are integer and contiguous
    df_tr[label_col] = df_tr[label_col].astype(int)
    df_val[label_col] = df_val[label_col].astype(int) if len(df_val) else df_val[label_col]
    df_te[label_col] = df_te[label_col].astype(int) if len(df_te) else df_te[label_col]
    uniq = sorted(pd.Series(df_tr[label_col]).unique().tolist())
    mapping = {v: i for i, v in enumerate(uniq)}
    df_tr[label_col] = df_tr[label_col].map(mapping).astype(int)
    if len(df_val):
        df_val[label_col] = df_val[label_col].map(mapping).astype(int)
    if len(df_te):
        df_te[label_col] = df_te[label_col].map(mapping).astype(int)
    n_choices = len(uniq)
    if obs_u_cols:
        obs_u_vals = df_tr[obs_u_cols].to_numpy(dtype=float)
        obs_u_rep = np.repeat(obs_u_vals[:, None, :], n_choices, axis=1)
        max_diff = float(np.max(np.abs(obs_u_rep - obs_u_rep[:, :1, :])))
        print(f"[biogeme] max |obs_u - obs_u_alt0| = {max_diff:.6f}")

    print("\n[diag] Chequeo de variacion de obs_u por alternativa")
    for c in obs_u_cols:
        vals = df_tr[c].to_numpy(dtype=float)
        max_diff = float(np.max(vals) - np.min(vals)) if len(vals) else 0.0
        print(f"  {c:30s} | range = {max_diff: .6f}")

    # Remove participant from modeling data (keep only for split)
    if args.participant_col in df_tr.columns:
        df_tr = df_tr.drop(columns=[args.participant_col])
    if args.participant_col in df_val.columns:
        df_val = df_val.drop(columns=[args.participant_col])
    if args.participant_col in df_te.columns:
        df_te = df_te.drop(columns=[args.participant_col])

    # Save filtered input for traceability
    args.results_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = args.results_dir / "biogeme_input_filtered.csv"
    df.to_csv(filtered_path, index=False)
    print(f"[biogeme] input saved: {filtered_path}")

    if df_tr is None or df_tr.empty:
        print("[warn] split train vacio; usando dataframe completo para estimacion.")
        df_tr = df
    database = db.Database("dataset_bicicletas_train", df_tr)

    # Variables
    Choice = Variable(label_col)
    obs_lt_vars = [Variable(c) for c in obs_lt_cols]
    obs_u_vars = [Variable(c) for c in obs_u_cols]
    obs_i_vars = [Variable(c) for c in obs_i_cols]

    # Latent variables: LV_k = Gamma_k * X + omega_k (Normal)
    if args.mnl_only:
        LVs = []
    else:
        gamma_betas = [
            [Beta(f"gamma_{k}_{c}", 0, None, None, 0) for c in obs_lt_cols]
            for k in range(args.n_latent)
        ]
        omegas = [Draws(f"omega_{k}", "NORMAL") for k in range(args.n_latent)]
        LVs = [
            sum(b * x for b, x in zip(gamma_betas[k], obs_lt_vars)) + omegas[k]
            for k in range(args.n_latent)
        ]

    # Utilities (base alt=0), per-alt betas to match ICLV
    V = {0: 0}
    beta_u_generic = [Beta(f"beta_{c}", 0, None, None, 0) for c in obs_u_cols]
    delta_generic = [Beta(f"delta_lv_{k}", 0, None, None, 0) for k in range(args.n_latent)]
    for alt in range(1, n_choices):
        asc = Beta(f"ASC_{alt}", 0, None, None, 0)
        util = asc + sum(b * x for b, x in zip(beta_u_generic, obs_u_vars))
        if LVs:
            util += sum(d * lv for d, lv in zip(delta_generic, LVs))
        V[alt] = util
    av = {k: 1 for k in V.keys()}
    P = models.logit(V, av, Choice)

    # Measurement: indicators ~ Normal(alpha + lambda * LV, sigma)
    meas_loglik = 0
    if not args.mnl_only and not args.no_measurement:
        for idx, y in enumerate(obs_i_vars):
            alpha = Beta(f"alpha_i{idx}", 0, None, None, 0)
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
            sigma = 1.0
            mu = alpha + sum(lam_k * lv for lam_k, lv in zip(lam, LVs))
            z = (y - mu) / sigma
            log_pdf = -0.5 * (np.log(2 * np.pi) + 2 * log(sigma) + z * z)
            meas_loglik += log_pdf

    if args.mnl_only:
        logprob = log(P)
    elif args.no_measurement:
        logprob = log(MonteCarlo(P))
    else:
        integrand = P * exp(meas_loglik)
        logprob = log(MonteCarlo(integrand))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.mnl_only:
        biogeme = bio.BIOGEME(database, logprob)
    else:
        biogeme = bio.BIOGEME(database, logprob, number_of_draws=args.n_draws)
    biogeme.algorithm = args.optimizer
    biogeme.model_name = args.model_name
    print(f"[biogeme] estimating model '{args.model_name}' (draws={args.n_draws}, optimizer={args.optimizer})")
    results = biogeme.estimate()
    print(f"[biogeme] estimation finished for '{args.model_name}'")
    try:
        est_debug = results.getEstimatedParameters()
        print("[diag] getEstimatedParameters head:")
        print(est_debug.head(5))
    except Exception as exc:
        print(f"[warn] getEstimatedParameters head failed: {exc}")

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
        ["get_pandas_estimated_parameters", "get_estimated_parameters", "getEstimatedParameters"],
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

    def _normalize_col(name: str) -> str:
        import re

        return re.sub(r"[^a-z0-9]+", "", name.lower())

    if est_params is not None:
        try:
            if isinstance(est_params, pd.DataFrame):
                est = est_params.copy()
            elif hasattr(est_params, "data"):
                est = pd.DataFrame(est_params.data)
            else:
                est = pd.DataFrame(est_params)
            if isinstance(est.index, pd.Index) and est.index.name:
                est = est.reset_index()
            est.columns = [c.lower().strip().replace(" ", "_") for c in est.columns]
            if "parameter" not in est.columns:
                if not isinstance(est.index, pd.RangeIndex):
                    est = est.reset_index().rename(columns={"index": "parameter"})
                for candidate in ["name", "param", "beta", "parameter_name"]:
                    if candidate in est.columns:
                        est = est.rename(columns={candidate: "parameter"})
                        break
            if "parameter" in est.columns:
                est = est.set_index("parameter")
                norm_map = {_normalize_col(c): c for c in est.columns}
                col_candidates = {
                    "beta": ["value", "estimate", "estimatedvalue"],
                    "std": ["stderr", "std_err", "standarderror", "robstderr", "robstderror", "robstd", "rob_std_err"],
                    "tstat": ["tstat", "t_test", "ttest", "tvalue", "robt_test", "robttest", "robttest", "rob_t_test"],
                    "pval": ["pval", "p_value", "pvalue", "robpval", "robpvalue", "rob_p_value"],
                }
                for dst, candidates in col_candidates.items():
                    for key in candidates:
                        if key in norm_map:
                            src = norm_map[key]
                            out[dst] = out[dst].fillna(out["name"].map(est[src]))
                            break
            else:
                print("[warn] estimated_parameters columns:", est.columns.tolist())
        except Exception as exc:
            print(f"[warn] failed to parse estimated_parameters: {exc}")

    for c in ["beta", "std", "tstat", "pval"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "stars" not in out.columns:
        out["stars"] = out["pval"].apply(stars_from_p)
    out_path = args.results_dir / "biogeme_params.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved params: {out_path}")
    print("General stats:", general)

    def _compute_metrics(split_name: str, split_df: pd.DataFrame) -> dict:
        if split_df is None or split_df.empty:
            return {}
        split_db = db.Database(f"dataset_bicicletas_{split_name}", split_df)
        probs = {f"p{alt}": models.logit(V, av, alt) for alt in V.keys()}
        prob_choice = models.logit(V, av, Choice)
        beta_values_sim = {k: v for k, v in beta_values.items() if not (k.startswith("alpha_i") or k.startswith("lambda_i"))}
        if args.mnl_only:
            sim = bio.BIOGEME(split_db, probs)
        else:
            sim = bio.BIOGEME(split_db, {k: MonteCarlo(v) for k, v in probs.items()}, number_of_draws=args.n_draws)
        sim.model_name = f"{args.model_name}_sim_{split_name}"
        sim_res = sim.simulate(beta_values_sim)
        p_mat = np.stack([sim_res[f"p{alt}"].to_numpy(dtype=float) for alt in sorted(V.keys())], axis=1)
        y = split_df[label_col].to_numpy(dtype=int)
        y_hat = np.argmax(p_mat, axis=1)
        acc = float(accuracy_score(y, y_hat))
        f1_macro = float(f1_score(y, y_hat, average="macro"))
        if n_choices == 2:
            f1_pos = float(f1_score(y, y_hat, pos_label=1))
            f1_neg = float(f1_score(y, y_hat, pos_label=0))
            auc = float(roc_auc_score(y, p_mat[:, 1])) if len(np.unique(y)) > 1 else float("nan")
        else:
            f1_pos = float("nan")
            f1_neg = float("nan")
            auc = float(roc_auc_score(y, p_mat, multi_class="ovr")) if len(np.unique(y)) > 1 else float("nan")
        if args.mnl_only:
            loglik = float(np.sum(np.log(np.clip(p_mat[np.arange(len(y)), y], 1e-9, 1))))
        else:
            prob_db = bio.BIOGEME(split_db, {"prob": MonteCarlo(prob_choice)}, number_of_draws=args.n_draws)
            prob_db.model_name = f"{args.model_name}_prob_{split_name}"
            prob_res = prob_db.simulate(beta_values_sim)
            p_choice = prob_res["prob"].to_numpy(dtype=float) if "prob" in prob_res.columns else prob_res.iloc[:, 0].to_numpy(dtype=float)
            loglik = float(np.sum(np.log(np.clip(p_choice, 1e-9, 1))))
        nll = float(-loglik)
        mean_nll = float(-loglik / max(1, len(y)))
        k = len(beta_values)
        aic = float(2 * k - 2 * loglik)
        bic = float(np.log(max(1, len(y))) * k - 2 * loglik)
        y_mean = y.mean()
        p_null = float(min(max(y_mean, 1e-9), 1 - 1e-9)) if n_choices == 2 else 1.0 / max(1, n_choices)
        if n_choices == 2:
            loglik_null = float(np.sum(y * np.log(p_null) + (1 - y) * np.log(1 - p_null)))
        else:
            loglik_null = float(len(y) * np.log(p_null))
        loglik_ratio = float(2 * (loglik - loglik_null))
        pseudo_r2 = float(1 - (loglik / loglik_null)) if loglik_null != 0 else float("nan")
        return {
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

    try:
        beta_values = get_with_fallback(results, ["get_beta_values", "getBetaValues"])
        if beta_values is None:
            raise RuntimeError("No se pudieron leer betas para simulacion.")
        metrics = {
            "train": _compute_metrics("train", df_tr),
            "val": _compute_metrics("val", df_val),
            "test": _compute_metrics("test", df_te),
        }
        metrics_path = args.results_dir / "biogeme_metrics.json"
        pd.Series(metrics).to_json(metrics_path, indent=2, force_ascii=False)
        print("Saved metrics:", metrics_path)
    except Exception as exc:
        print(f"[warn] no se pudieron calcular metricas predictivas: {exc}")


if __name__ == "__main__":
    main()
