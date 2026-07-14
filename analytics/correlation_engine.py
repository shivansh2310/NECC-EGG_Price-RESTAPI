from __future__ import annotations

import pandas as pd


def price_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert long-form API prices into a date-by-market matrix."""
    if prices.empty:
        return pd.DataFrame()
    return (
        prices.pivot_table(index="date", columns="market", values="price", aggfunc="last")
        .sort_index()
        .ffill(limit=7)
    )


def return_correlations(prices: pd.DataFrame, minimum_observations: int = 30) -> pd.DataFrame:
    """Pearson correlations of daily percentage changes between markets."""
    matrix = price_matrix(prices)
    if matrix.empty:
        return matrix
    return matrix.pct_change(fill_method=None).corr(min_periods=minimum_observations)


def rolling_correlation(
    prices: pd.DataFrame, market_a: str, market_b: str, window: int = 30
) -> pd.DataFrame:
    matrix = price_matrix(prices)
    if market_a not in matrix or market_b not in matrix:
        return pd.DataFrame(columns=["date", "correlation"])
    returns = matrix[[market_a, market_b]].pct_change(fill_method=None)
    values = returns[market_a].rolling(window, min_periods=max(5, window // 3)).corr(returns[market_b])
    return values.rename("correlation").reset_index()

