"""Evaluation metric helpers."""

import numpy as np
import pandas as pd

def r_squared(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Coefficient of determination R² = 1 - SS_res / SS_tot."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)

def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Fraction of predictions where sign matches the actual sign."""
    yt = np.sign(np.asarray(y_true))
    yp = np.sign(np.asarray(y_pred))
    return float((yt == yp).mean())
