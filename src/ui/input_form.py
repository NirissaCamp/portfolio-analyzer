"""Streamlit input widget for building a Portfolio."""

import io
import pandas as pd
import streamlit as st
from src.data.models import Holding, Portfolio

def render_portfolio_input() -> Portfolio | None:
    """Render holdings input. Returns a Portfolio or None if empty."""
    mode = st.radio("Input mode", ["Manual entry", "Upload CSV"], horizontal=True)

    if mode == "Upload CSV":
        return _csv_input()
    return _manual_input()


def _manual_input() -> Portfolio | None:
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
    return _build_portfolio(edited, default_name="My portfolio")


def _csv_input() -> Portfolio | None:
    st.subheader("Upload CSV")
    st.caption("Required columns: ticker, shares, cost_basis")
    uploaded = st.file_uploader("Choose CSV", type=["csv"])
    if uploaded is None:
        return None

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")
        return None

    required = {"ticker", "shares", "cost_basis"}
    if not required.issubset(df.columns):
        st.error(f"CSV must have columns: {required}")
        return None

    rows = df.to_dict(orient="records")
    return _build_portfolio(rows, default_name="Uploaded Portfolio")


def _build_portfolio(rows: list[dict], default_name: str) -> Portfolio | None:
    holdings: list[Holding] = []
    skipped = 0
    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            skipped += 1
            continue
        try:
            shares = float(row.get("shares") or 0)
            cost = float(row.get("cost_basis") or 0)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if shares <= 0:
            skipped += 1
            continue
        holdings.append(Holding(ticker=ticker, shares=shares, cost_basis=cost))

    if skipped > 0:
        st.warning(f"Skipped {skipped} invalid row(s)")
    if not holdings:
        return None
    name = st.text_input("Portfolio name", value=default_name)
    return Portfolio(name=name, holdings=holdings)
