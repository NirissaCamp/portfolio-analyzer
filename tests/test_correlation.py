"""Tests for correlation matrix."""

import math
import pandas as pd
from src.analytics.correlation import correlation_matrix


def _series(values: list[float], start_date: str = "2024-01-02") -> pd.Series:
    idx = pd.bdate_range(start=start_date, periods=len(values))
    return pd.Series(values, index=idx)


def test_correlation_diagonal_is_one():
    aapl = _series([100.0, 102.0, 104.0, 103.0])
    msft = _series([200.0, 198.0, 202.0, 205.0])
    result = correlation_matrix({"AAPL": aapl, "MSFT": msft})
    assert math.isclose(result.loc["AAPL", "AAPL"], 1.0)
    assert math.isclose(result.loc["MSFT", "MSFT"], 1.0)


def test_correlation_identical_series_is_one():
    s = _series([100.0, 102.0, 104.0, 103.0])
    result = correlation_matrix({"A": s, "B": s})
    assert math.isclose(result.loc["A", "B"], 1.0)


def test_correlation_perfectly_negative():
    s1 = _series([100.0, 110.0, 100.0, 110.0])
    s2 = _series([100.0, 90.0, 100.0, 90.0])
    result = correlation_matrix({"X": s1, "Y": s2})
    assert math.isclose(result.loc["X", "Y"], -1.0, abs_tol=1e-9)
