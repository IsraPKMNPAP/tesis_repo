from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, bioDraws, MonteCarlo, log, exp


def load_cols(path: Path) -> list[str]:
    return [c.strip().lower() for c in path.read_text().splitlines() if c.strip()]


def warn_missing(df: pd.DataFrame, cols: list[str], label: str) -> list[str]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[warn] {label} missing cols: {missing}")
    return [c for c in cols if c in df.columns]


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

    database = db.Database("neuma", df)

    # Variables
    Choice = Variable(label_col)
    obs_lt_vars = [Variable(c) for c in obs_lt_cols]
    obs_u_vars = [Variable(c) for c in obs_u_cols]
    obs_i_vars = [Variable(c) for c in obs_i_cols]

    # Latent variable: LV = Gamma * X + omega (Normal)
    gamma_betas = [Beta(f"gamma_{c}", 0, None, None, 0) for c in obs_lt_cols]
    omega = bioDraws("omega", "NORMAL")
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
    biogeme = bio.BIOGEME(database, logprob, numberOfDraws=args.n_draws)
    biogeme.modelName = "icl_v_biogeme"
    results = biogeme.estimate()

    general = results.getGeneralStatistics()
    betas = results.getBetaValues()
    std = results.getBetaStdErrors()
    tstat = results.getTTest()
    pval = results.getPValues()

    out = pd.DataFrame(
        [
            {
                "name": name,
                "beta": betas[name],
                "std": std[name],
                "tstat": tstat[name],
                "pval": pval[name],
            }
            for name in betas.keys()
        ]
    )

    out_path = args.results_dir / "biogeme_params.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved params: {out_path}")
    print("General stats:", general)


if __name__ == "__main__":
    main()
