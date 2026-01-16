from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.features import load_features_file


def preprocess_utility(df: pd.DataFrame, cols: List[str], cat_unique_threshold: int) -> tuple[np.ndarray, List[str]]:
    df_prep = df[cols].copy()
    cat_cols: List[str] = []
    num_cols: List[str] = []
    for c in df_prep.columns:
        if df_prep[c].dtype == object or df_prep[c].nunique(dropna=True) <= cat_unique_threshold:
            cat_cols.append(c)
        else:
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


def logit_metrics(y: np.ndarray, prob: np.ndarray, k: int) -> dict:
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    ll = (y * np.log(prob) + (1 - y) * np.log(1 - prob)).sum()
    n = len(y)
    aic = 2 * k - 2 * ll
    bic = np.log(n) * k - 2 * ll
    p0 = y.mean()
    p0 = min(max(p0, 1e-6), 1 - 1e-6)
    ll_null = (y * np.log(p0) + (1 - y) * np.log(1 - p0)).sum()
    llr = 2 * (ll - ll_null)
    return {
        "log_likelihood": float(ll),
        "aic": float(aic),
        "bic": float(bic),
        "loglik_null": float(ll_null),
        "loglik_ratio": float(llr),
        "n": int(n),
        "k": int(k),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparacion logit: utilidad vs dataset gigante.")
    parser.add_argument("--utility-data", type=Path, required=True)
    parser.add_argument("--obs-u-cols", type=Path, required=True)
    parser.add_argument("--joined-data", type=Path, required=True)
    parser.add_argument("--label-col", type=str, default="bought")
    parser.add_argument("--cat-unique-threshold", type=int, default=4)
    args = parser.parse_args()

    label = args.label_col.lower()

    # Utility logit (MNL sin latentes)
    df_u = pd.read_csv(args.utility_data)
    df_u.columns = df_u.columns.str.lower()
    obs_u_cols = [c.strip().lower() for c in load_features_file(args.obs_u_cols)]
    obs_u_cols = [c for c in obs_u_cols if c in df_u.columns]
    if not obs_u_cols:
        raise ValueError("No se encontraron columnas obs_u en utility-data.")
    y_u = pd.to_numeric(df_u[label], errors="coerce").to_numpy(dtype=int)
    X_u, _ = preprocess_utility(df_u, obs_u_cols, args.cat_unique_threshold)
    clf_u = LogisticRegression(penalty="l2", C=1e6, solver="liblinear", max_iter=1000)
    clf_u.fit(X_u, y_u)
    prob_u = clf_u.predict_proba(X_u)[:, 1]
    k_u = X_u.shape[1] + 1
    m_u = logit_metrics(y_u, prob_u, k_u)

    # Joined dataset logit (EEG+tab+img)
    df_j = pd.read_csv(args.joined_data)
    df_j.columns = df_j.columns.str.lower()
    if label not in df_j.columns:
        raise ValueError(f"No se encontro label '{label}' en joined-data.")
    keep_cols = [c for c in df_j.columns if c != label and "bought" not in c]
    X = df_j[keep_cols]
    y = df_j[label].astype(int).to_numpy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    X_prep = preprocessor.fit_transform(X)
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(X_prep, y)
    prob = clf.predict_proba(X_prep)[:, 1]
    k = X_prep.shape[1] + 1
    m_j = logit_metrics(y, prob, k)

    print("Utility logit metrics:", m_u)
    print("Joined logit metrics:", m_j)


if __name__ == "__main__":
    main()
