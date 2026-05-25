"""Tests for Portfolio and Holding dataclasses."""

import pytest
from src.data.models import Holding, Portfolio

def test_holding_basic_construction():
    h = Holding(ticker="AAPL", shares=10.0, cost_basis=150.0)
    assert h.ticker == "AAPL"
    assert h.shares == 10.0
    assert h.cost_basis == 150.0

def test_portfolio_with_multiple_holdings():
    p = Portfolio(
        name="Tech Mix",
        holdings = [
            Holding(ticker="AAPL", shares=10.0, cost_basis=150.0),
            Holding(ticker="MSFT", shares=5.0,  cost_basis=300.0),
        ],
    )
    assert p.name == "Tech Mix"
    assert len(p.holdings) == 2
    assert p.holdings[0].ticker == "AAPL"

def test_empty_portfolio_allowed():
    p = Portfolio(name="Empty", holdings=[])
    assert p.holdings == []
