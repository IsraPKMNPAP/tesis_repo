from __future__ import annotations

from typing import Iterable, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer


def features_labels(df: pd.DataFrame, features: Iterable[str], label: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[list(features)].copy()
    y = df[label].copy()
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
):
    y_strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y_strat)


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list[str], list[str]]:
    numeric = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical = X.select_dtypes(include=["category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return preprocessor, numeric, categorical


def encode_labels(y_train: pd.Series, y_test: pd.Series):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    return le, y_train_enc, y_test_enc

