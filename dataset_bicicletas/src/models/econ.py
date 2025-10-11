from __future__ import annotations

import numpy as np
import statsmodels.api as sm


def fit_mnlogit(X_proc: np.ndarray, y_enc: np.ndarray):
    X_with_const = sm.add_constant(X_proc, has_constant="add")
    model = sm.MNLogit(y_enc, X_with_const)
    res = model.fit(method="newton", maxiter=100, disp=False)
    return res

