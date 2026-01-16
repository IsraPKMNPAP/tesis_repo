from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils.features import load_features_file


def preprocess_utility(df: pd.DataFrame, cols: List[str], cat_unique_threshold: int) -> tuple[np.ndarray, List[str]]:
    df_prep = df[cols].copy()
    cat_cols: List[str] = []
    num_cols: List[str] = []
    for c in df_prep.columns:
        if df_prep[c].dtype == object:
            cat_cols.append(c)
        else:
            try:
                if df_prep[c].nunique(dropna=True) <= cat_unique_threshold:
                    cat_cols.append(c)
                else:
                    num_cols.append(c)
            except Exception:
                num_cols.append(c)

    parts = []
    names: List[str] = []
    if num_cols:
        num_vals = df_prep[num_cols].apply(pd.to_numeric, errors="coerce")
        num_vals = num_vals.fillna(num_vals.median())
        scaler = StandardScaler()
        num_scaled = scaler.fit_transform(num_vals)
        parts.append(num_scaled)
        names.extend(num_cols)
    if cat_cols:
        cat_df = df_prep[cat_cols].astype(str)
        cat_dummies = pd.get_dummies(cat_df, drop_first=True, prefix=[f"{c}" for c in cat_cols])
        parts.append(cat_dummies.to_numpy(dtype=float))
        names.extend(cat_dummies.columns.tolist())

    if not parts:
        return np.zeros((len(df_prep), 0)), []
    X = np.concatenate(parts, axis=1)
    return X, names


def wald_stats(X: np.ndarray, y: np.ndarray, coef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Fisher information for logit: X^T W X
    logits = X @ coef
    p = 1.0 / (1.0 + np.exp(-logits))
    w = p * (1.0 - p)
    W = np.diag(w)
    fisher = X.T @ W @ X
    try:
        cov = np.linalg.pinv(fisher)
    except Exception:
        cov = np.linalg.pinv(fisher + 1e-6 * np.eye(fisher.shape[0]))
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    tstat = coef / std
    return std, tstat


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Logit sobre variables de utilidad (obs_u) con estadisticos.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--obs-u-cols", type=Path, required=True)
    parser.add_argument("--label-col", type=str, default="bought")
    parser.add_argument("--cat-unique-threshold", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=0, help="Limitar filas para debug (0 = todas).")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.lower()
    label_col = args.label_col.lower()
    if label_col not in df.columns:
        raise ValueError(f"No se encontro etiqueta '{label_col}'.")

    obs_u_cols = [c.strip().lower() for c in load_features_file(args.obs_u_cols)]
    obs_u_cols = [c for c in obs_u_cols if c in df.columns]
    if not obs_u_cols:
        raise ValueError("No se encontraron columnas obs_u en el dataset.")

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    y = pd.to_numeric(df[label_col], errors="coerce").to_numpy(dtype=int)
    X, names = preprocess_utility(df, obs_u_cols, cat_unique_threshold=args.cat_unique_threshold)
    if X.shape[1] == 0:
        raise ValueError("X quedo sin columnas luego del preprocesamiento.")

    # add intercept
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    names = ["intercept"] + names

    clf = LogisticRegression(penalty="l2", C=1e6, solver="liblinear", max_iter=1000)
    clf.fit(X, y)
    coef = np.concatenate([clf.intercept_, clf.coef_.ravel()])

    std, tstat = wald_stats(X_design, y, coef)
    rows = []
    for name, b, s, t in zip(names, coef, std, tstat):
        rows.append({"feature": name, "coef": b, "std": s, "tstat": t, "stars": stars_for_t(t)})
    out = pd.DataFrame(rows)
    # Metrics: loglik, AIC/BIC, LLR vs null
    prob = clf.predict_proba(X)[:, 1]
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    ll = (y * np.log(prob) + (1 - y) * np.log(1 - prob)).sum()
    k = len(coef)
    n = len(y)
    aic = 2 * k - 2 * ll
    bic = np.log(n) * k - 2 * ll
    p0 = y.mean()
    p0 = min(max(p0, 1e-6), 1 - 1e-6)
    ll_null = (y * np.log(p0) + (1 - y) * np.log(1 - p0)).sum()
    llr = 2 * (ll - ll_null)

    print(out.to_string(index=False))
    print("Metrics:", {"log_likelihood": float(ll), "aic": float(aic), "bic": float(bic), "loglik_null": float(ll_null), "loglik_ratio": float(llr)})


if __name__ == "__main__":
    main()
