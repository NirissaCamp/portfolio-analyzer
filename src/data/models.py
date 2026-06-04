"""Data classes for portfolio holdings."""

from dataclasses import dataclass

@dataclass
class Holding:
    """A single position: how many shares of one ticker at what cost basis."""
    ticker: str
    shares: float
    cost_basis: float

@dataclass
class Portfolio:
    """A user-named collection of holdings."""
    name:str
    holdings: list[Holding]
