"""Portfolio Analyzer - Streamlit entry point."""

from datetime import date, timedelta
import streamlit as st
import plotly.graph_objects as go

from src.data import get_price_history

st.set_page_config(page_title="Portfolio Analyzer", layout="wide")

def main() -> None:
    st.title("Portfolio Analyzer")
    st.caption("MVP - single ticker only for now")

    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    days_back = st.slider("Lookback (days)", min_value=30, max_value=730, value=365)

    if not ticker:
        st.info("Enter a ticker to begin")
        return

    end = date.today()
    start = end - timedelta(days=days_back)

    with st.spinner(f"Fetching {ticker}..."):
        prices = get_price_history(ticker, start, end)
    if prices.empty:
        st.error(f"No data found for {ticker}")
        return

    st.success(f"Loaded {len(prices)} days of {ticker} data")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices["Close"], name=ticker))
    fig.update_layout(title = f"{ticker} Close Price", height=400)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
