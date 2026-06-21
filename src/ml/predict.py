"""Model loading and inference for the Forecast tab."""

import math
from pathlib import Path
import joblib
import pandas as pd

from src.config import LINEAR_MODEL_PATH, XGBOOST_MODEL_PATH

FEATURE_COLUMNS: list[str] = [
    "return_1d", "return_5d", "return_20d",
    "volatility_20d", "sma_ratio_5_20", "rsi_14",
    "volume_ratio_20d", "market_return_5d",
]

def load_model(path: Path):
    """Load a serialized model from disk. Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

def load_models() -> dict:
    """Load both Linear and XGBoost models."""
    return{
        "linear": load_model(LINEAR_MODEL_PATH),
        "xgboost": load_model(XGBOOST_MODEL_PATH),
    }

def predict_for_features(model, features_row: pd.DataFrame) -> float:
    """Predict a 5-day return for a single feature row.

    Returns NaN if the input contains NaN (we don't propagate the
    risk of sklearn raising downstream).
    """
    if features_row[FEATURE_COLUMNS].isna().any().any():
        return float("nan")
    pred = model.predict(features_row[FEATURE_COLUMNS])
    return float(pred[0])
