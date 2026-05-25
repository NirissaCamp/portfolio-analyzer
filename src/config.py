"""Project-wide constants. Imported by data, analytics, and UI layers."""

from pathlib import Path

# Financial constants
RISK_FREE_RATE: float = 0.04   # 4% annual, approximates US T-bill yield
TRADING_DAYS_PER_YEAR: int = 252
BENCHMARK_TICKER: str = "^GSPC" #S&P 500 ticker on yfinance
DEFAULT_LOOKBACK: str = "ly"    #yfinance period string

#Storage
CACHE_DB_PATH: Path = Path("data")/"cache.db"
