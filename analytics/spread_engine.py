from __future__ import annotations

import pandas as pd

from analytics.correlation_engine import price_matrix


def market_spread(
    prices: pd.DataFrame, market_a: str, market_b: str, z_window: int = 30
) -> pd.DataFrame:
    """Calculate A-minus-B spread and its rolling z-score."""
    matrix = price_matrix(prices)
    if market_a not in matrix or market_b not in matrix:
        return pd.DataFrame(columns=["date", "spread", "z_score"])
    spread = (matrix[market_a] - matrix[market_b]).dropna().rename("spread")
    mean = spread.rolling(z_window, min_periods=max(5, z_window // 3)).mean()
    std = spread.rolling(z_window, min_periods=max(5, z_window // 3)).std()
    result = pd.DataFrame({"spread": spread, "z_score": (spread - mean) / std.replace(0, pd.NA)})
    return result.reset_index()

