from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.correlation_engine import price_matrix


def _rolling_z_score(series: pd.Series, window: int = 365) -> pd.Series:
    minimum = min(60, max(10, window // 4))
    mean = series.rolling(window, min_periods=minimum).mean()
    std = series.rolling(window, min_periods=minimum).std()
    return (series - mean) / std.replace(0, np.nan)


def market_stress_index(prices: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Composite stress score from volatility, dispersion, decoupling and divergence."""
    matrix = price_matrix(prices)
    if matrix.empty:
        return pd.DataFrame()
    returns = matrix.pct_change(fill_method=None)
    national_return = returns.mean(axis=1)

    volatility = national_return.rolling(window, min_periods=7).std()
    dispersion = matrix.std(axis=1) / matrix.mean(axis=1).replace(0, np.nan)
    divergence = returns.sub(national_return, axis=0).abs().mean(axis=1).rolling(window, min_periods=7).mean()

    rolling_correlation = []
    for position in range(len(returns)):
        sample = returns.iloc[max(0, position - window + 1) : position + 1]
        if len(sample) < 7:
            rolling_correlation.append(np.nan)
            continue
        corr = sample.corr().to_numpy()
        upper = corr[np.triu_indices_from(corr, k=1)]
        rolling_correlation.append(np.nanmean(upper) if np.isfinite(upper).any() else np.nan)
    decoupling = 1 - pd.Series(rolling_correlation, index=matrix.index)

    components = pd.DataFrame(
        {
            "volatility": volatility,
            "dispersion": dispersion,
            "decoupling": decoupling,
            "divergence": divergence,
        }
    )
    standardized = components.apply(_rolling_z_score).clip(-3, 3)
    composite = standardized.mean(axis=1, skipna=False)
    score = composite.rolling(365, min_periods=60).rank(pct=True) * 100
    result = components.assign(score=score)
    result["regime"] = pd.cut(
        result["score"], bins=[-np.inf, 60, 80, np.inf], labels=["Normal", "Watch", "Stress"]
    ).astype(object)
    return result.reset_index(names="date")

