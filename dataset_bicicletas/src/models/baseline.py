from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def build_logistic_pipeline(preprocessor) -> Pipeline:
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000,
    )
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    return pipe


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor,
) -> Tuple[Pipeline, str, np.ndarray]:
    pipe = build_logistic_pipeline(preprocessor)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred)
    probs = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
    return pipe, report, probs

