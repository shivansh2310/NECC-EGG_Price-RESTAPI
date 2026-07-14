from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    forecast: pd.DataFrame
    metrics: dict[str, float]
    model: str


def prepare_series(prices: pd.DataFrame, market: str | None = None) -> pd.Series:
    """Create a continuous daily series from API price records."""
    frame = prices
    if market is not None:
        frame = frame[frame["market"] == market]
    if frame.empty:
        return pd.Series(dtype=float)
    series = frame.groupby("date")["price"].mean().sort_index()
    series.index = pd.DatetimeIndex(series.index)
    return series.asfreq("D").interpolate(limit=7).ffill().dropna().astype(float)


def _predict(train: pd.Series, horizon: int, model: str) -> np.ndarray:
    if model == "7-day seasonal naive":
        season = train.tail(min(7, len(train))).to_numpy()
        return np.resize(season, horizon)

    try:
        if model == "Exponential smoothing":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            seasonal = "add" if len(train) >= 28 else None
            period = 7 if seasonal else None
            fitted = ExponentialSmoothing(
                train,
                trend="add",
                damped_trend=True,
                seasonal=seasonal,
                seasonal_periods=period,
                initialization_method="estimated",
            ).fit(optimized=True)
            return np.asarray(fitted.forecast(horizon))
        if model == "ARIMA":
            from statsmodels.tsa.arima.model import ARIMA

            fitted = ARIMA(train, order=(2, 1, 2)).fit()
            return np.asarray(fitted.forecast(horizon))
    except (ImportError, ValueError, np.linalg.LinAlgError) as exc:
        raise RuntimeError(f"{model} could not be fitted: {exc}") from exc
    raise ValueError(f"Unknown model: {model}")


def error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    nonzero = actual != 0
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE": float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100) if nonzero.any() else np.nan,
    }


def forecast_prices(
    series: pd.Series,
    horizon: int = 30,
    model: str = "Exponential smoothing",
    test_days: int = 30,
) -> ForecastResult:
    """Backtest a model, then refit it to produce a future forecast."""
    minimum = max(test_days + 14, 35)
    if len(series) < minimum:
        raise ValueError(f"At least {minimum} daily observations are required.")

    train, test = series.iloc[:-test_days], series.iloc[-test_days:]
    backtest = _predict(train, len(test), model)
    metrics = error_metrics(test.to_numpy(), backtest)
    future = _predict(series, horizon, model)

    residual_std = float(np.std(test.to_numpy() - backtest, ddof=1))
    steps = np.arange(1, horizon + 1)
    uncertainty = 1.96 * residual_std * np.sqrt(steps)
    dates = pd.date_range(series.index[-1] + pd.offsets.Day(1), periods=horizon, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "forecast": future,
            "lower": np.maximum(0, future - uncertainty),
            "upper": future + uncertainty,
        }
    )
    return ForecastResult(frame, metrics, model)
