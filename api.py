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
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CSV_PATH = Path("necc_egg_prices_daily.csv")
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
