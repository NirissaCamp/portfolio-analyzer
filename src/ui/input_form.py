"""Streamlit input widget for building a Portfolio from manual entry."""

import streamlit as st
from src.data.models import Holding, Portfolio

def render_portfolio_input() -> Portfolio | None:
    """Render a data editor for holdings. Returns a Portfolio or None if empty."""
    st.subheader("Holdings")

    default_rows = [
        {"ticker": "AAPL", "shares":10.0, "cost_basis":150.0},
        {"ticker": "MSFT", "shares":5.0,  "cost_basis":300.0},
    ]
    edited = st.data_editor(
        default_rows,
        num_rows="dynamic",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", required=True),
            "shares": st.column_config.NumberColumn("Shares", min_value=0.0, step=1.0),
            "cost_basis": st.column_config.NumberColumn("Cost Basis ($)", min_value=0.0),
        },
        key="holdings_editor",
    )

    holdings = []
    for row in edited:
        ticker = (row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        shares = float(row.get("shares") or 0)
        cost = float(row.get("cost_basis") or 0)
        if shares <= 0:
            continue
        holdings.append(Holding(ticker=ticker, shares=shares, cost_basis=cost))
    if not holdings:
        return None
    name = st.text_input("Portfolio name", value="My Portfolio")
    return Portfolio(name=name, holdings=holdings)
