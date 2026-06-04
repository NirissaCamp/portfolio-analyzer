"""Fetch historical price data from Yahoo Finance via yfinance."""

from datetime import date
import pandas as pd
import yfinance as yf

def fetch_prices(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Fetch daily OHLCV data for a ticker.
    Returns a DataFrame with DatetimeIndex and columns including 'Close'.
    Returns an empty DataFrame if the ticker is invalid or data is missing.
    """
    data = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if data.empty:
        return data
    # Normalize index to date-only (yfinance returns timezone-aware datetimes)
    data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
    return data
