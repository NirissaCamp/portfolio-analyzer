"""Project-wide constants. Imported by data, analytics, and UI layers."""

from pathlib import Path

# Financial constants
RISK_FREE_RATE: float = 0.04   # 4% annual, approximates US T-bill yield
TRADING_DAYS_PER_YEAR: int = 252
BENCHMARK_TICKER: str = "^GSPC" #S&P 500 ticker on yfinance
DEFAULT_LOOKBACK: str = "ly"    #yfinance period string

#Storage
CACHE_DB_PATH: Path = Path("data")/"cache.db"

# Phase 2: ML forecasting
N_FORECAST_DAYS: int = 5              #predict 5 days ahead
TRAINING_START: str = "2021-01-01"    #for scripts/train.py
TRAINING_END:  str = "2026-01-01"
TRAIN_TEST_CUTOFF:str = "2025-01-01"  #chronological split point
MODEL_DIR:  Path = Path("models")
LINEAR_MODEL_PATH: Path = MODEL_DIR / "linear.pkl"
XGBOOST_MODEL_PATH: Path = MODEL_DIR / "xgboost.pkl"
TRAINING_LOG_PATH: Path = MODEL_DIR / "training_log.txt"

# Top 50 S&P 500 tickers by market cap (snapshot as of project start)
TRAINING_UNIVERSE: list[str] = [
      "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "TSLA", "LLY", "AVGO",
      "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "ORCL", "HD",
      "COST", "BAC", "MRK", "ABBV", "CVX", "KO", "ADBE", "PEP", "CRM", "WFC",
      "TMO", "MCD", "CSCO", "ABT", "ACN", "DHR", "LIN", "AMD", "TXN", "INTU",
      "NKE", "DIS", "PM", "VZ", "CMCSA", "IBM", "QCOM", "NEE", "AMGN", "T",
]
