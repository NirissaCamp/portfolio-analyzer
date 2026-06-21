"""Tests for the prediction pipeline."""

import pandas as pd
import pytest
import joblib
from pathlib import Path
from src.ml.predict import predict_for_features, FEATURE_COLUMNS


def test_feature_columns_are_eight():
    assert len(FEATURE_COLUMNS) == 8

def test_predict_for_features_returns_float(tmp_path):
    """Train a tiny Ridge model and verify  predict_for_features returns a float."""
    from sklearn.linear_model import Ridge
    import numpy as np

    #Train a toy model: y = sum of features
    n = 100
    rng = np.random.default_rng(42)
    X_train = pd.DataFrame(
        rng.random((n, 8)),
        columns=FEATURE_COLUMNS,
    )
    y_train = X_train.sum(axis=1)
    model = Ridge()
    model.fit(X_train, y_train)

    #Predict on a single new row
    features_row = pd.DataFrame([[0.5]*8], columns=FEATURE_COLUMNS)
    result = predict_for_features(model, features_row)
    assert isinstance(result, float)

def test_predict_for_features_handles_nan_row():
    """If features contain NaN, predicct_for_features returns NaN (not crash)."""
    from sklearn.linear_model import Ridge
    import numpy as np

    n = 100
    X_train = pd.DataFrame(
        np.random.default_rng(42).random((n, 8)),
        columns=FEATURE_COLUMNS,
    )
    y_train = X_train.sum(axis=1)
    model = Ridge().fit(X_train, y_train)

    features_row = pd.DataFrame([[0.5, float("nan")] + [0.5]*6], columns=FEATURE_COLUMNS)
    result = predict_for_features(model, features_row)
    #Some sklearn versions raise on NaN, ours should detect and return NaN
    import math
    assert math.isnan(result)
