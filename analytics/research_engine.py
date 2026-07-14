from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.correlation_engine import price_matrix


def principal_components(prices: pd.DataFrame, components: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PCA of standardized market returns via singular-value decomposition."""
    returns = price_matrix(prices).pct_change(fill_method=None).dropna(how="all")
    valid = returns.columns[returns.notna().sum() >= 30]
    clean = returns[valid].fillna(0)
    if clean.empty or clean.shape[1] < 2:
        return pd.DataFrame(), pd.DataFrame()
    standardized = (clean - clean.mean()) / clean.std().replace(0, np.nan)
    standardized = standardized.fillna(0)
    _, singular, vectors = np.linalg.svd(standardized.to_numpy(), full_matrices=False)
    count = min(components, len(singular), clean.shape[1])
    variance = singular**2 / max(len(standardized) - 1, 1)
    ratio = variance / variance.sum()
    summary = pd.DataFrame(
        {
            "component": [f"PC{i}" for i in range(1, count + 1)],
            "explained_variance": ratio[:count],
            "cumulative_variance": np.cumsum(ratio[:count]),
        }
    )
    loadings = pd.DataFrame(
        vectors[:count].T,
        index=clean.columns,
        columns=summary["component"],
    ).reset_index(names="market")
    return summary, loadings


def cointegration_test(prices: pd.DataFrame, market_a: str, market_b: str) -> dict[str, float | bool]:
    """Engle-Granger test for a stable long-run relationship between price levels."""
    from statsmodels.tsa.stattools import coint

    matrix = price_matrix(prices)[[market_a, market_b]].dropna()
    if len(matrix) < 60:
        raise ValueError("At least 60 overlapping observations are required.")
    statistic, p_value, _ = coint(matrix[market_a], matrix[market_b])
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "cointegrated": bool(p_value < 0.05),
    }


def structural_breaks(
    series: pd.Series, window: int = 60, threshold: float = 3.0, minimum_gap: int = 30
) -> pd.DataFrame:
    """Flag separated shifts where short-term and prior-window means diverge."""
    if len(series) < window * 2:
        return pd.DataFrame(columns=["date", "score", "before", "after"])
    prior = series.shift(window).rolling(window, min_periods=window).mean()
    recent = series.rolling(window, min_periods=window).mean()
    # Scale the level shift by recent day-to-day movement. This intentionally
    # measures practical displacement rather than a classical t-statistic.
    scale = series.diff().rolling(window * 2, min_periods=window).std()
    score = ((recent - prior) / scale.replace(0, np.nan)).abs()
    candidates = score[score >= threshold].sort_values(ascending=False)
    selected: list[pd.Timestamp] = []
    for candidate in candidates.index:
        if all(abs((candidate - existing).days) >= minimum_gap for existing in selected):
            selected.append(candidate)
    selected.sort()
    return pd.DataFrame(
        {
            "date": selected,
            "score": [score.loc[item] for item in selected],
            "before": [prior.loc[item] for item in selected],
            "after": [recent.loc[item] for item in selected],
        }
    )


def volatility_regimes(series: pd.Series, window: int = 30) -> pd.DataFrame:
    """Classify rolling volatility relative to its full-history distribution."""
    volatility = series.pct_change(fill_method=None).rolling(window, min_periods=7).std() * np.sqrt(365)
    available = volatility.dropna()
    if available.empty:
        return pd.DataFrame(columns=["date", "volatility", "regime"])
    low, high = available.quantile([0.33, 0.67])
    regime = pd.cut(
        volatility,
        bins=[-np.inf, low, high, np.inf],
        labels=["Low", "Normal", "High"],
    )
    return pd.DataFrame({"date": series.index, "volatility": volatility, "regime": regime.astype(object)})
