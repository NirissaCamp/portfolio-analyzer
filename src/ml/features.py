"""Feature engineering for  return forecasting."""

import pandas as pd

def return_lag(prices: pd.Series, n: int) -> pd.Series:
    """Return the n-period lookback return for each date.
    return_lag(t, n) = prices[t] / prices[t-n] - 1
    First n rows are NaN (no prior price).
    """
    return prices / prices.shift(n) - 1

def volatility(prices: pd.Series, window: int=20) -> pd.Series:
    """Rolling standard deviation of daily returns over 'window' days."""
    daily_returns = prices.pct_change()
    return daily_returns.rolling(window).std(ddof=0)

def sma_ratio(prices: pd.Series, short: int=5, long: int=20) -> pd.Series:
    """Ratio of short-window SMA to long-window SMA.
    > 1 means short-term trend above long-term; < 1 the reverse.
    """
    return prices.rolling(short).mean() / prices.rolling(long).mean()

def volume_ratio(volume: pd.Series, window: int=20) -> pd.Series:
    """Today's volume divided by rolling average volume."""
    return volume / volume.rolling(window).mean()

def rsi(prices: pd.Series, window: int=14) -> pd.Series:
    """Relative Strength Index (RSI) - 14 days classic momentum oscillator.
    RSI ranges 0-100. > 70 typically "overbought"; < 30 typically "oversold".
    Uses simple moving average of gains/losses (not Wilder's smoothing).
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = - delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def build_features(prices: pd.Series, volumes: pd.Series, market_prices: pd.Series) -> pd.DataFrame:
    """Build the 8-feature DataFrame for a single ticker.
    Each row is one date; columns are the 8 features defined in the spec.
    Returns a DataFrame indexed by date; early rows will contain NaN
    (warmup period for moving averages.)
    """
    return pd.DataFrame({
        "return_1d": return_lag(prices, 1),
        "return_5d": return_lag(prices, 5),
        "return_20d": return_lag(prices, 20),
        "volatility_20d": volatility(prices, 20),
        "sma_ratio_5_20": sma_ratio(prices, 5, 20),
        "rsi_14": rsi(prices, 14),
        "volume_ratio_20d": volume_ratio(volumes, 20),
        "market_return_5d": return_lag(market_prices, 5),
    })

def build_dataset(ticker_data: dict[str, dict[str, pd.Series]],
                  market_prices: pd.Series,
                  forecast_days: int = 5,)->tuple[pd.DataFrame, pd.Series]:
    """Build a training dataset from multiple tickers.
    Args:
    ticker_data: dict{ticker: {"Close": pd.Series, "Volume": pd.Series}}
    market_prices: S&P 500 close prices indexed by date
    forecast_days: how many days ahead the target return represents
    Returns:
        X: DataFrame of features (one row per (ticker, date)), no NaN rows
        y: Series of forward returns aligned with X
    """
    feature_frames = []
    target_series = []
    for ticker, data in ticker_data.items():
        prices = data["Close"]
        volumes = data["Volume"]

        features = build_features(prices, volumes, market_prices)
        features["ticker"] = ticker

        #Forward return: (price_t+n /price_t) - 1
        target = prices.shift(-forecast_days) / prices - 1

        #Align features with target, drop NaN rows
        combined = features.copy()
        combined["target"] = target
        combined = combined.dropna()

        feature_frames.append(combined.drop(columns=["target"]))
        target_series.append(combined["target"])

    X = pd.concat(feature_frames, axis=0)
    y = pd.concat(target_series, axis=0)
    return X, y
