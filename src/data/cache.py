"""SQLite-backed cache for daily price data."""

import sqlite3
from datetime import date, datetime
from pathlib import Path
import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS price_cache(
    ticker TEXT NOT NULL,
    date   DATE NOT NULL,
    close  REAL NOT NULL,
    fetched_at TIMESTAMP NOT NULL,
    PRIMARY KEY (ticker, date)
);
"""

def init_cache(db_path: Path) -> None:
    """Create the schema if it doesn't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)

def put_prices(db_path: Path, ticker:str, df: pd.DataFrame) -> None:
    """Insert or replace price rows for a ticker."""
    init_cache(db_path)
    now = datetime.now().isoformat()
    rows = [
        (ticker, idx.date().isoformat(), float(row["Close"]), now)
        for idx, row in df.iterrows()
    ]
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO price_cache VALUES(?, ?, ?, ?)",
            rows,
        )

def get_prices(db_path: Path, ticker: str, start: date, end: date) -> pd.DataFrame:
    """Read cached prices for a ticker between dates(inclusive)."""
    init_cache(db_path)
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT date, close FROM price_cache "
            "WHERE ticker = ? AND date BETWEEN ? AND ? ORDER BY date",
            conn,
            params=(ticker, start.isoformat(), end.isoformat()),
            parse_dates=["date"],
        )
    if df.empty:
        return df
    df = df.set_index("date").rename(columns={"close": "Close"})
    return df

def is_fresh(db_path: Path, ticker: str) -> bool:
    """Return True if cache for this ticker was updated today."""
    init_cache(db_path)
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT MAX(fetched_at) FROM price_cache WHERE ticker = ?",
            (ticker,),
        )
        latest = cur.fetchone()[0]
    if latest is None:
        return False
    return datetime.fromisoformat(latest).date() == date.today()
