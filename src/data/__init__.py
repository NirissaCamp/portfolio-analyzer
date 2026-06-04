"""Data layer: combines cache lookups with yfinance fetches."""

from datetime import date
from pathlib import Path
import pandas as pd

from src.config import CACHE_DB_PATH
from src.data.cache import get_prices as _cache_get, put_prices, is_fresh
from src.data.fetcher import fetch_prices

def get_price_history(
        ticker: str,
        start: date,
        end: date,
        db_path: Path = CACHE_DB_PATH,
)-> pd.DataFrame:
    """Return daily prices for 'ticker' between dates.
    Tries the local  cache first. On miss or staleness, fetches from yfinance
    and  writes through to the cache.
    """
    if is_fresh(db_path, ticker):
        cached = _cache_get(db_path, ticker, start, end)
        if not cached.empty:
            return cached

    fresh = fetch_prices(ticker, start, end)
    if not fresh.empty:
        put_prices(db_path, ticker, fresh)
    return fresh[["Close"]] if not fresh.empty else fresh
