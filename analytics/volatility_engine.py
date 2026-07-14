from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.correlation_engine import price_matrix


def rolling_volatility(prices: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Annualized rolling volatility for each market."""
    matrix = price_matrix(prices)
    if matrix.empty:
        return pd.DataFrame(columns=["date", "market", "volatility"])
    returns = matrix.pct_change(fill_method=None)
    volatility = returns.rolling(window, min_periods=max(5, window // 3)).std() * np.sqrt(365)
    return volatility.rename_axis(columns="market").stack().rename("volatility").reset_index()


def volatility_ranking(prices: pd.DataFrame, lookback: int = 90) -> pd.DataFrame:
    matrix = price_matrix(prices)
    if matrix.empty:
        return pd.DataFrame(columns=["market", "volatility"])
    values = matrix.tail(lookback).pct_change(fill_method=None).std() * np.sqrt(365)
    return values.dropna().sort_values(ascending=False).rename("volatility").reset_index()

