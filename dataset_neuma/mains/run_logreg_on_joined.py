from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Logistic regression sobre joined_eeg_multimodal_latente.csv")
    parser.add_argument("--input-csv", type=Path, default=Path("./data/processed/joined_eeg_multimodal_latente.csv"))
    parser.add_argument("--label-col", type=str, default="bought")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df.columns = df.columns.str.lower()
    label_col = args.label_col.lower()
    if label_col not in df.columns:
        raise SystemExit(f"No se encontró label {label_col} en {args.input_csv}")

    # Features = todo excepto label
    keep_cols = [c for c in df.columns if c != label_col and "bought" not in c]
    X = df[keep_cols]
    y = df[label_col].astype(int).to_numpy()

    # Separar numéricas y categóricas
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("logreg", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, (prob >= 0.5).astype(int), digits=3))


if __name__ == "__main__":
    main()
