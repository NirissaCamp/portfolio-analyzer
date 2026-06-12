"""One-shot training pipeline.
Run with:
    python scripts/train.py

Fetches historical data for the training universe, builds features,
trains Ridge + XGBoost, evaluates on a chronological hold-out split,
saves models as .pkl, appends results to training_log.txt.
"""

import time
from datetime import date, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from src.config import(
    BENCHMARK_TICKER,
    LINEAR_MODEL_PATH,
    MODEL_DIR,
    N_FORECAST_DAYS,
    TRAINING_END,
    TRAINING_LOG_PATH,
    TRAINING_START,
    TRAINING_UNIVERSE,
    TRAIN_TEST_CUTOFF,
    XGBOOST_MODEL_PATH,
)
from src.data.fetcher import fetch_prices
from src.ml.evaluate import directional_accuracy, mae, r_squared
from src.ml.features import build_dataset

def fetch_universe(ticker: list[str], start: date, end: date, sleep_s: float = 1.0)->dict[str, dict[str, pd.Series]]:
    """Fetch close + volume for many tickers, sleeping between requests."""
    result = {}
    for i, ticker in enumerate(ticker, 1):
        print(f"  [{i}/{len(ticker)}] {ticker}", flush=True)
        try:
            df = fetch_prices(ticker, start, end)
            if df.empty:
                print(f"   skipped - no data")
                continue
            result[ticker] = {"Close": df["Close"], "Volume": df["Volume"]}
        except Exception as e:
            print(f"   skipped - {e}")
        time.sleep(sleep_s)    #avoid rate limit
    return result

def chronological_split(X: pd.DataFrame, y: pd.Series, cutoff: str
                        )-> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split by date: dates < cutoff in train, dates >= cutoff in test."""
    cutoff_ts = pd.Timestamp(cutoff)
    train_mask = X.index < cutoff_ts
    return X[train_mask], y[train_mask], X[~train_mask], y[~train_mask]

def evaluate_and_log(model_name: str, model, X_test, y_test) -> str:
    """Compute metrics and return a log line."""
    y_pred = model.predict(X_test.drop(columns=["ticker"], errors="ignore"))
    r2 = r_squared(y_test, y_pred)
    err = mae(y_test, y_pred)
    da = directional_accuracy(y_test, y_pred)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"{timestamp} | {model_name:<8} | R²={r2:.4f} MAE={err:.4f} DirAcc={da:.4f}"
    print(line)
    return line


def main():
    print("=" * 60)
    print("Phase 2 Training Pipeline")
    print("=" * 60)

    start = datetime.strptime(TRAINING_START, "%Y-%m-%d").date()
    end = datetime.strptime(TRAINING_END, "%Y-%m-%d").date()

    print(f"\n[1/5] Fetching {len(TRAINING_UNIVERSE)} tickers ({start} → {end})...")
    ticker_data = fetch_universe(TRAINING_UNIVERSE, start, end)
    print(f"       Got {len(ticker_data)}/{len(TRAINING_UNIVERSE)} tickers successfully")

    print(f"\n[2/5] Fetching benchmark {BENCHMARK_TICKER}...")
    bench_df = fetch_prices(BENCHMARK_TICKER, start, end)
    market_prices = bench_df["Close"]

    print(f"\n[3/5] Building features and target ...")
    X, y = build_dataset(ticker_data, market_prices, forecast_days=N_FORECAST_DAYS)
    print(f"        Total samples: {len(X)}")

    print(f"\n[4/5] Splitting at {TRAIN_TEST_CUTOFF}...")
    X_train, y_train, X_test, y_test = chronological_split(X, y, TRAIN_TEST_CUTOFF)
    print(f"        Train: {len(X_train)} | Test: {len(X_test)}")

    # Drop the 'ticker' column from features fed to models (it's categorical metadata)
    feature_cols = [c for c in X_train.columns if c != "ticker"]

    print(f"\n[5/5] Training models...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("  Ridge...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train[feature_cols], y_train)
    joblib.dump(ridge, LINEAR_MODEL_PATH)
    ridge_line = evaluate_and_log("Linear", ridge, X_test, y_test)

    print("  XGBoost...")
    xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                       subsample=0.8, random_state=42,)
    xgb.fit(X_train[feature_cols], y_train)
    joblib.dump(xgb,XGBOOST_MODEL_PATH)
    xgb_line = evaluate_and_log("XGBoost", xgb, X_test, y_test)

    print(f"\nSaved: {LINEAR_MODEL_PATH}")
    print(f"Saved: {XGBOOST_MODEL_PATH}")

    #Append to training log
    with open(TRAINING_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(ridge_line + "\n")
        f.write(xgb_line + "\n")
    print(f"Appended to: {TRAINING_LOG_PATH}")


if __name__ == "__main__":
    main()
