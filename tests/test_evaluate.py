"""Test for evaluation metric helpers."""
import math
import numpy as np
import pandas as pd
from src.ml.evaluate import r_squared, mae, directional_accuracy

def test_r_squared_perfect_prediction():
    y_true = pd.Series([0.01, 0.02, -0.01, 0.03, 0.02])
    y_pred = y_true.copy()
    assert math.isclose(r_squared(y_true, y_pred), 1.0)

def test_r_squared_mean_prediction_is_zero():
    """Predicting the mean for all samples -> R² ≈ 0."""
    y_true = pd.Series([0.01, -0.01, 0.02, -0.02, 0.03])
    y_pred = pd.Series([y_true.mean()] * len(y_true))
    assert math.isclose(r_squared(y_true, y_pred), 0.0, abs_tol=1e-9)

def test_mae_basic():
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.5, 2.0, 2.5])
    #errors: 0.5, 0.0, 0.5 -> mean 0.333...
    assert math.isclose(mae(y_true, y_pred), 1.0/3.0)

def test_directional_accuracy_all_correct():
    y_true = pd.Series([0.01, -0.02, 0.03])
    y_pred = pd.Series([0.05, -0.01, 0.10]) # signs all match
    assert math.isclose(directional_accuracy(y_true, y_pred), 1.0)

def test_directional_accuracy_half_correct():
    y_true = pd.Series([0.01, -0.01, 0.01, -0.01])
    y_pred = pd.Series([0.01, 0.01, -0.01, -0.01])
    assert math.isclose(directional_accuracy(y_true, y_pred), 0.5)
