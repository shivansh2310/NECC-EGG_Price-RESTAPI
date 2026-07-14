import asyncio
import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from analytics.correlation_engine import (
    price_matrix,
    return_correlations,
    rolling_correlation,
)
from analytics.forecasting_engine import forecast_prices
from analytics.index_engine import latest_market_snapshot, national_index
from analytics.market_metadata import market_metadata
from analytics.network_engine import centrality_table, correlation_network, lead_lag_network
from analytics.research_engine import (
    cointegration_test,
    principal_components,
    structural_breaks,
    volatility_regimes,
)
from analytics.spread_engine import market_spread
from analytics.stress_engine import market_stress_index
from analytics.volatility_engine import rolling_volatility, volatility_ranking

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Resolve beside this module so the API works regardless of the process cwd.
CSV_PATH = Path(__file__).resolve().with_name("necc_egg_prices_daily.csv")
REQUIRED_COLUMNS = {
    "date",
    "market",
    "category",
    "price",
    "price_filled",
}

REFRESH_TIME = os.environ.get("REFRESH_TIME", "09:00")


def load_prices(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        raise RuntimeError(
            f"{csv_path} was not found. Run scraper.py before starting the API."
        )

    loaded = pd.read_csv(csv_path, parse_dates=["date"])
    missing_columns = REQUIRED_COLUMNS.difference(loaded.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RuntimeError(f"{csv_path} is missing required columns: {missing}")

    loaded["price"] = pd.to_numeric(loaded["price"], errors="coerce")
    loaded["price_filled"] = pd.to_numeric(loaded["price_filled"], errors="coerce")
    return loaded.sort_values(["date", "category", "market"]).reset_index(drop=True)


def reload_data(app: FastAPI) -> None:
    app.state.df = load_prices()
    app.state.last_refreshed = datetime.now()
    logger.info("Data reloaded — %d rows", len(app.state.df))


def seconds_until_next_run(refresh_time: str) -> float:
    hour, minute = map(int, refresh_time.split(":"))
    now = datetime.now()
    target = datetime.combine(now.date(), time(hour, minute))
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def run_scraper() -> None:
    logger.info("Running scraper...")
    result = subprocess.run(
        [sys.executable, "scraper.py", "--output", str(CSV_PATH)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Scraper failed:\n%s", result.stderr)
    else:
        logger.info("Scraper completed successfully")


async def daily_refresh_loop(app: FastAPI) -> None:
    while True:
        wait = seconds_until_next_run(REFRESH_TIME)
        logger.info("Next refresh at %s (in %.0f seconds)", REFRESH_TIME, wait)
        await asyncio.sleep(wait)
        try:
            run_scraper()
            reload_data(app)
            app.state.next_refresh = REFRESH_TIME
        except Exception as exc:
            logger.error("Refresh failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CSV_PATH.exists():
        logger.info("CSV not found — running initial scrape (full history)...")
        run_scraper()

    reload_data(app)
    app.state.next_refresh = REFRESH_TIME

    task = asyncio.create_task(daily_refresh_loop(app))
    yield
    task.cancel()


app = FastAPI(title="NECC Egg Prices API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def to_json_records(frame: pd.DataFrame) -> list[dict]:
    cleaned = frame.astype(object).where(pd.notna(frame), None)
    return cleaned.to_dict(orient="records")


def filter_prices(
    df: pd.DataFrame,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    market: Optional[str] = None,
    category: Optional[str] = None,
    use_filled: bool = False,
) -> pd.DataFrame:
    filtered = df.copy()
    if start_date:
        filtered = filtered[filtered["date"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered = filtered[filtered["date"] <= pd.to_datetime(end_date)]
    if market:
        filtered = filtered[
            filtered["market"].str.contains(market, case=False, na=False, regex=False)
        ]
    if category:
        filtered = filtered[filtered["category"].str.upper() == category.upper()]

    price_col = "price_filled" if use_filled else "price"
    result = filtered[["date", "market", "category", price_col]].copy()
    result.rename(columns={price_col: "price"}, inplace=True)
    return result


@app.get("/")
def root(request: Request):
    return {
        "service": "NECC Egg Prices API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health(request: Request):
    df = request.app.state.df
    return {
        "status": "ok",
        "rows": len(df),
        "min_date": df["date"].min().date().isoformat(),
        "max_date": df["date"].max().date().isoformat(),
        "last_refreshed": request.app.state.last_refreshed.isoformat(),
        "next_refresh": request.app.state.next_refresh,
    }


@app.get("/prices/date-range")
def get_date_range(request: Request):
    df = request.app.state.df
    return {
        "min_date": df["date"].min().date().isoformat(),
        "max_date": df["date"].max().date().isoformat(),
    }

@app.get("/prices/markets")
def get_markets(request: Request):
    df = request.app.state.df
    return {"markets": sorted(df["market"].unique().tolist())}

@app.get("/prices/categories")
def get_categories(request: Request):
    df = request.app.state.df
    return {"categories": sorted(df["category"].unique().tolist())}

@app.get("/prices")
def get_prices(
    request: Request,
    start_date: Annotated[Optional[date], Query(description="YYYY-MM-DD")] = None,
    end_date: Annotated[Optional[date], Query(description="YYYY-MM-DD")] = None,
    market: Annotated[
        Optional[str],
        Query(description="Filter by market name (substring)"),
    ] = None,
    category: Annotated[Optional[str], Query(description="NECC or PREVAILING")] = None,
    use_filled: Annotated[
        bool,
        Query(description="Use interpolated prices (price_filled)"),
    ] = False,
    limit: Annotated[
        int,
        Query(ge=1, le=10000, description="Max rows to return"),
    ] = 100,
    offset: Annotated[int, Query(ge=0, description="Skip rows for pagination")] = 0,
):
    df = request.app.state.df
    filtered = filter_prices(df, start_date, end_date, market, category, use_filled)
    total = len(filtered)
    filtered = filtered.iloc[offset:offset+limit]

    filtered["date"] = filtered["date"].dt.date.astype(str)

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": to_json_records(filtered),
    }

@app.get("/prices/stats")
def get_stats(
    request: Request,
    start_date: Annotated[Optional[date], Query(description="YYYY-MM-DD")] = None,
    end_date: Annotated[Optional[date], Query(description="YYYY-MM-DD")] = None,
    market: Annotated[
        Optional[str],
        Query(description="Filter by market name (substring)"),
    ] = None,
    category: Annotated[Optional[str], Query(description="NECC or PREVAILING")] = None,
    use_filled: Annotated[bool, Query(description="Use interpolated prices")] = False,
):
    df = request.app.state.df
    filtered = filter_prices(df, start_date, end_date, market, category, use_filled)
    if filtered.empty:
        return {"stats": []}

    stats = (
        filtered.groupby(["market", "category"])["price"]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
        .round(2)
    )
    return {"stats": to_json_records(stats)}


@app.post("/admin/refresh")
def admin_refresh(request: Request):
    run_scraper()
    reload_data(request.app)
    return {
        "status": "ok",
        "message": "Data refreshed successfully",
        "rows": len(request.app.state.df),
        "last_refreshed": request.app.state.last_refreshed.isoformat(),
    }


def _necc_prices(request: Request, start_date=None, end_date=None) -> pd.DataFrame:
    df = request.app.state.df
    df = df[df["category"] == "NECC"].copy()
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    if df.empty:
        raise HTTPException(status_code=404, detail="No NECC data for the given period")
    return df[["date", "market", "category", "price_filled"]].rename(
        columns={"price_filled": "price"}
    )


@app.get("/analytics/metadata")
def analytics_metadata():
    return {"markets": to_json_records(market_metadata())}


@app.get("/analytics/index")
def analytics_index(
    request: Request,
    start_date: Annotated[Optional[date], Query()] = None,
    end_date: Annotated[Optional[date], Query()] = None,
    weighted: Annotated[bool, Query(description="Population-weighted index")] = False,
):
    prices = _necc_prices(request, start_date, end_date)
    weights = None
    if weighted:
        meta = market_metadata().set_index("market")["population_weight"]
        weights = meta.to_dict()
    idx = national_index(prices, weights)
    if idx.empty:
        return {"index": [], "latest": None}
    idx["date"] = idx["date"].dt.date.astype(str)
    latest = idx.iloc[-1].to_dict()
    latest["date"] = latest["date"]
    return {"index": to_json_records(idx), "latest": latest}


@app.get("/analytics/snapshot")
def analytics_snapshot(
    request: Request,
    start_date: Annotated[Optional[date], Query()] = None,
    end_date: Annotated[Optional[date], Query()] = None,
):
    prices = _necc_prices(request, start_date, end_date)
    snap = latest_market_snapshot(prices)
    if snap.empty:
        return {"snapshot": []}
    return {"snapshot": to_json_records(snap)}


@app.get("/analytics/correlation")
def analytics_correlation(
    request: Request,
    start_date: Annotated[Optional[date], Query()] = None,
    end_date: Annotated[Optional[date], Query()] = None,
):
    prices = _necc_prices(request, start_date, end_date)
    corr = return_correlations(prices)
    if corr.empty:
        return {"correlation": []}
    return {"markets": corr.columns.tolist(), "matrix": corr.values.tolist()}


@app.get("/analytics/rolling-correlation")
def analytics_rolling_correlation(
    request: Request,
    market_a: Annotated[str, Query()],
    market_b: Annotated[str, Query()],
    window: Annotated[int, Query(ge=5, le=365)] = 30,
    start_date: Annotated[Optional[date], Query()] = None,
    end_date: Annotated[Optional[date], Query()] = None,
):
    prices = _necc_prices(request, start_date, end_date)
    rc = rolling_correlation(prices, market_a, market_b, window)
    if rc.empty:
        return {"rolling_correlation": []}
    rc["date"] = rc["date"].dt.date.astype(str)
    return {"rolling_correlation": to_json_records(rc)}


@app.get("/analytics/volatility")
def analytics_volatility(
    request: Request,
    window: Annotated[int, Query(ge=5, le=365)] = 90,
    start_date: Annotated[Optional[date], Query()] = None,
    end_date: Annotated[Optional[date], Query()] = None,
):
    prices = _necc_prices(request, start_date, end_date)
    ranking = volatility_ranking(prices, window)
    if ranking.empty:
        return {"volatility_ranking": []}
    return {"volatility_ranking": to_json_records(ranking)}


@app.get("/analytics/rolling-volatility")
def analytics_rolling_volatility(
    request: Request,
    window: Annotated[int, Query(ge=5, le=365)] = 30,
    start_date: Annotated[Optional[date], Query()] = None,
    end_date: Annotated[Optional[date], Query()] = None,
):
    prices = _necc_prices(request, start_date, end_date)
    rv = rolling_volatility(prices, window)
    if rv.empty:
        return {"rolling_volatility": []}
    rv["date"] = rv["date"].dt.date.astype(str)
    return {"rolling_volatility": to_json_records(rv)}


@app.get("/analytics/spread")
def analytics_spread(
    request: Request,
    market_a: Annotated[str, Query()],
    market_b: Annotated[str, Query()],
    z_window: Annotated[int, Query(ge=10, le=365)] = 90,
    start_date: Annotated[Optional[date], Query()] = None,
    end_date: Annotated[Optional[date], Query()] = None,
):
    prices = _necc_prices(request, start_date, end_date)
    sp = market_spread(prices, market_a, market_b, z_window)
    if sp.empty:
        return {"spread": []}
    sp["date"] = sp["date"].dt.date.astype(str)
    return {"spread": to_json_records(sp)}


@app.get("/analytics/forecast")
def analytics_forecast(
    request: Request,
    market: Annotated[str, Query()],
    horizon: Annotated[int, Query(ge=1, le=30)] = 7,
    model: Annotated[str, Query(description="seasonal_naive, exponential_smoothing, arima")] = "seasonal_naive",
    start_date: Annotated[Optional[date], Query()] = None,
    end_date: Annotated[Optional[date], Query()] = None,
):
    prices = _necc_prices(request, start_date, end_date)
    model_map = {
        "seasonal_naive": "7-day seasonal naive",
        "exponential_smoothing": "Exponential smoothing",
        "arima": "ARIMA",
    }
    model_name = model_map.get(model)
    if model_name is None:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Use: {', '.join(model_map.keys())}")
    from analytics.forecasting_engine import prepare_series
    series = prepare_series(prices, market)
    if series.empty:
        raise HTTPException(status_code=404, detail=f"Market '{market}' not found")
    try:
        result = forecast_prices(series, horizon, model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    forecast_df = result.forecast
    forecast_df["date"] = forecast_df["date"].dt.date.astype(str)
    return {
        "market": market,
        "model": result.model,
        "metrics": result.metrics,
        "forecast": to_json_records(forecast_df),
    }


@app.get("/analytics/network")
def analytics_network(
    request: Request,
    kind: Annotated[str, Query(description="correlation or lead_lag")] = "correlation",
    threshold: Annotated[float, Query(ge=0.0, le=1.0)] = 0.5,
    lag: Annotated[int, Query(ge=1, le=10)] = 1,
    start_date: Annotated[Optional[date], Query()] = None,
    end_date: Annotated[Optional[date], Query()] = None,
):
    prices = _necc_prices(request, start_date, end_date)
    if kind == "lead_lag":
        graph = lead_lag_network(prices, lag, threshold)
    else:
        graph = correlation_network(prices, threshold)
    nodes = [{"id": n, **graph.nodes[n]} for n in graph.nodes]
    edges = [{"source": u, "target": v, **graph.edges[u, v]} for u, v in graph.edges]
    centrality = centrality_table(graph)
    return {
        "nodes": nodes,
        "edges": edges,
        "centrality": to_json_records(centrality),
    }


@app.get("/analytics/stress")
def analytics_stress(
    request: Request,
    window: Annotated[int, Query(ge=10, le=365)] = 90,
    start_date: Annotated[Optional[date], Query()] = None,
    end_date: Annotated[Optional[date], Query()] = None,
):
    prices = _necc_prices(request, start_date, end_date)
    stress = market_stress_index(prices, window)
    if stress.empty:
        return {"stress": []}
    stress["date"] = stress["date"].dt.date.astype(str)
    return {"stress": to_json_records(stress)}


@app.get("/analytics/pca")
def analytics_pca(
    request: Request,
    start_date: Annotated[Optional[date], Query()] = None,
    end_date: Annotated[Optional[date], Query()] = None,
):
    prices = _necc_prices(request, start_date, end_date)
    variance, loadings = principal_components(prices)
    if variance.empty:
        return {"variance_explained": [], "loadings": []}
    return {
        "variance_explained": to_json_records(variance),
        "loadings": to_json_records(loadings),
    }


@app.get("/analytics/cointegration")
def analytics_cointegration(
    request: Request,
    market_a: Annotated[str, Query()],
    market_b: Annotated[str, Query()],
    start_date: Annotated[Optional[date], Query()] = None,
    end_date: Annotated[Optional[date], Query()] = None,
):
    prices = _necc_prices(request, start_date, end_date)
    result = cointegration_test(prices, market_a, market_b)
    return result


@app.get("/analytics/structural-breaks")
def analytics_structural_breaks(
    request: Request,
    market: Annotated[str, Query()],
    window: Annotated[int, Query(ge=10, le=365)] = 60,
    threshold: Annotated[float, Query(ge=0.5, le=5.0)] = 2.0,
    start_date: Annotated[Optional[date], Query()] = None,
    end_date: Annotated[Optional[date], Query()] = None,
):
    prices = _necc_prices(request, start_date, end_date)
    series = price_matrix(prices)
    if market not in series.columns:
        raise HTTPException(status_code=404, detail=f"Market '{market}' not found")
    s = series[market].dropna()
    breaks = structural_breaks(s, window, threshold)
    if not breaks.empty:
        breaks["date"] = breaks["date"].dt.date.astype(str)
    return {"market": market, "breaks": to_json_records(breaks)}


@app.get("/analytics/volatility-regimes")
def analytics_volatility_regimes(
    request: Request,
    market: Annotated[str, Query()],
    window: Annotated[int, Query(ge=10, le=365)] = 60,
    start_date: Annotated[Optional[date], Query()] = None,
    end_date: Annotated[Optional[date], Query()] = None,
):
    prices = _necc_prices(request, start_date, end_date)
    series = price_matrix(prices)
    if market not in series.columns:
        raise HTTPException(status_code=404, detail=f"Market '{market}' not found")
    s = series[market].dropna()
    regimes = volatility_regimes(s, window)
    regimes["date"] = regimes["date"].dt.date.astype(str)
    return {"market": market, "regimes": to_json_records(regimes)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
