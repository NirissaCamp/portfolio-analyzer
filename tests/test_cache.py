from datetime import date, datetime, timedelta
import pandas as pd
import pytest
from src.data.cache import init_cache, put_prices, get_prices, is_fresh

@pytest.fixture
def temp_db(tmp_path):
    """Provide a fresh SQLite path for each other."""
    db_gtipath = tmp_path / "test_cache.db"
    init_cache(db_path)
    return db_path

def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
    )

def test_put_then_get_returns_same_data(temp_db):
      df = _sample_df()
      put_prices(temp_db, "AAPL", df)
      result = get_prices(temp_db, "AAPL", date(2025, 1, 2), date(2025, 1, 6))
      assert len(result) == 3
      assert result["Close"].tolist() == [100.0, 101.0, 102.0]


def test_get_unknown_ticker_returns_empty(temp_db):
      result = get_prices(temp_db, "NOSUCH", date(2025, 1, 1), date(2025, 1, 10))
      assert result.empty


def test_is_fresh_true_for_today(temp_db):
      df = _sample_df()
      put_prices(temp_db, "AAPL", df)
      assert is_fresh(temp_db, "AAPL") is True


def test_is_fresh_false_for_no_data(temp_db):
      assert is_fresh(temp_db, "MISSING") is False
