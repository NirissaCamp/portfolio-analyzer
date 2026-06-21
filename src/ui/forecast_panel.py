"""Streamlit UI for the Forecast tab.
Renders the 4 sections described in the design spec:
1. Portfolio-level forecast (2 big metric cards)
2. Pre-holding prediction table
3. Model comparison bar chart
4. Model metadata footer
"""
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import(
    BENCHMARK_TICKER,
    LINEAR_MODEL_PATH,
    TRAINING_LOG_PATH,
    XGBOOST_MODEL_PATH,
)
from src.data import get_price_history
from src.data.models import Portfolio
from src.ml.features import build_features
from src.ml.predict import FEATURE_COLUMNS, predict_for_features


@st.cache_resource    #这行代码的作用是什么？
def _load_models() -> dict:
    """Cached model loader (loads once per process)."""
    return{
        "linear": joblib.load(LINEAR_MODEL_PATH),
        "xgboost": joblib.load(XGBOOST_MODEL_PATH),
    }


def _read_log_tail(n_lines: int = 2) -> list[str]:
    """Read the last N lines of training_log.txt."""
    if not TRAINING_LOG_PATH.exists():
        return []
    with open(TRAINING_LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return lines[-n_lines:] if lines else []


def _compute_features_row(ticker: str, end: date, market_prices: pd.Series) -> pd.DataFrame | None:
    """Build the latest 1-row feature vector for a ticker."""
    start = end - timedelta(days=90)
    df = get_price_history(ticker, start, end)
    if df.empty or len(df) < 30:
        return None

    #Need Volume too - re-fetch via fetcher directly (since get_price_history returns Close only)
    from src.data.fetcher import fetch_prices
    full = fetch_prices(ticker, start, end)
    if full.empty:
        return None
    features = build_features(full["Close"], full["Volume"], market_prices)
    #Drop rows where any feature is NaN (warmup period + index misalignment at the tail)
    features = features.dropna()
    if features.empty:
        return None
    return features.iloc[[-1]]


def render_forecast_tab(portfolio: Portfolio) -> None:
    """Top-level entry point. Called from app.py inside 'with tab2'."""
    st.info(
        "Predictions are 5-day forward returns based on technical features. "
        "Trained on S&P 500 top 50, 2021-2024."
    )

    if portfolio is None or not portfolio.holdings:
        st.warning("Add holdings in the Analysis tab to see forecasts.")
        return

    #Load medels (cached)
    try:
        models = _load_models()
    except FileNotFoundError:
        st.error(
            "Model files not found. Run 'python scripts/train.py' first, "
            "then commit 'models/*.pkl'."
        )
        return

    #Fetch market benchmark for the market_return_5d feature
    end = date.today()
    start = end - timedelta(days=90)
    bench_df = get_price_history(BENCHMARK_TICKER, start, end)
    if bench_df.empty:
        st.error("Could not load S&P 500 benchmark for feature computation.")
        return
    market_prices = bench_df["Close"]

    #Per-holding predictions
    rows = []
    for h in portfolio.holdings:
        features_row = _compute_features_row(h.ticker, end, market_prices)
        if features_row is None or features_row.isna().any().any():
            st.warning(f"Insufficient or invalid data for {h.ticker}, skipping.")
            continue
        linear_pred = predict_for_features(models["linear"], features_row)
        xgb_pred = predict_for_features(models["xgboost"], features_row)
        df = get_price_history(h.ticker, end - timedelta(days=10), end)
        if df.empty:
            continue
        current_price = float(df["Close"].iloc[-1])
        agreement = "✅" if (linear_pred > 0) == (xgb_pred > 0) else "⚠️"
        rows.append({
            "Ticker": h.ticker,
            "Current Price": f"${current_price:,.2f}",
            "Linear pred": f"{linear_pred:+2%}",
            "XGB pred": f"{xgb_pred:+.2%}",
            "Agreement": agreement,
            "_linear_raw": linear_pred,
            "_xgb_raw":xgb_pred,
            "_market_value": current_price * h.shares,
        })

    if not rows:
        st.error("No forecasts could be computed.")
        return

    # Section 1 : portfolio-level forecast
    st.subheader("📊 Portfolio-level 5-day Forecast")
    total_value = sum(r["_market_value"] for r in rows)
    linear_portfolio = sum(r["_linear_raw"] * r["_market_value"] for r in rows) / total_value
    xgb_portfolio = sum(r["_xgb_raw"] * r["_market_value"] for r in rows) / total_value
    col1, col2 = st.columns(2)
    col1.metric("Linear forecast", f"{linear_portfolio:+.2%}")
    col2.metric("XGBoost forecast", f"{xgb_portfolio:+.2%}")
    st.caption("Expected 5-day return based on current holdings, weighted by market value.")

    st.divider()

    # Section 2: pre-holding table
    st.subheader("🔍 Pre-Holding Predictions")
    table_df = pd.DataFrame(rows)[["Ticker", "Current Price", "Linear pred", "XGB pred", "Agreement"]]
    st.dataframe(table_df, hide_index=True, use_container_width=True)

    st.divider()

    # Section 3: comparison bar chart
    st.subheader("📈 Model Comparison")
    fig = go.Figure()
    tickers = [r["Ticker"] for r in rows]
    fig.add_trace(go.Bar(
        y=tickers, x=[r["_linear_raw"] for r in rows],
        name="Linear", orientation="h",
    ))
    fig.add_trace(go.Bar(
        y=tickers, x=[r["_xgb_raw"] for r in rows],
        name="XGBoost", orientation="h",
    ))
    fig.update_layout(
        barmode="group",
        xaxis_title="Predicted 5-day return",
        height=max(300, 60 * len(tickers)),
        xaxis_tickformat=".1%"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Section 4: model metadata footer
    st.subheader("ℹ️ Model Metadata")
    log_lines = _read_log_tail(2)
    if log_lines:
        for line in log_lines:
            st.code(line, language=None)
    else:
        st.caption("No training log found. Run 'Python scripts/train.py' to generate.")
