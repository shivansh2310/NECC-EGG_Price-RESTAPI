from __future__ import annotations

import numpy as np
import pandas as pd


def national_index(
    prices: pd.DataFrame, weights: dict[str, float] | None = None
) -> pd.DataFrame:
    """Build an equal- or custom-weighted daily index, rebased to 100."""
    if prices.empty:
        return pd.DataFrame(columns=["date", "price", "index", "return", "volatility"])
    if weights is None:
        daily = prices.groupby("date", as_index=False)["price"].mean()
    else:
        weighted = prices.assign(weight=prices["market"].map(weights)).dropna(subset=["weight"])
        if weighted.empty:
            return pd.DataFrame(columns=["date", "price", "index", "return", "volatility"])
        weighted["weighted_price"] = weighted["price"] * weighted["weight"]
        totals = weighted.groupby("date", as_index=False).agg(
            weighted_price=("weighted_price", "sum"), weight=("weight", "sum")
        )
        totals["price"] = totals["weighted_price"] / totals["weight"]
        daily = totals[["date", "price"]]
    daily = daily.sort_values("date")
    first = daily["price"].dropna().iloc[0]
    daily["index"] = daily["price"] / first * 100
    daily["return"] = daily["price"].pct_change()
    daily["volatility"] = daily["return"].rolling(30, min_periods=7).std() * np.sqrt(365)
    return daily


def latest_market_snapshot(prices: pd.DataFrame) -> pd.DataFrame:
    """Return each market's latest price with 1, 7 and 30-day changes."""
    if prices.empty:
        return pd.DataFrame(columns=["market", "price", "1D", "7D", "30D"])
    pivot = prices.pivot_table(index="date", columns="market", values="price", aggfunc="last").sort_index()
    output = []
    for market in pivot.columns:
        series = pivot[market].dropna()
        if series.empty:
            continue
        row = {"market": market, "price": series.iloc[-1]}
        for days, label in ((1, "1D"), (7, "7D"), (30, "30D")):
            row[label] = series.pct_change(days).iloc[-1] * 100 if len(series) > days else np.nan
        output.append(row)
    return pd.DataFrame(output).sort_values("1D", ascending=False, na_position="last")
