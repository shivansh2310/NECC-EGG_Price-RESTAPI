from datetime import date
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# Resolve beside this module so the API works regardless of the process cwd.
CSV_PATH = Path(__file__).resolve().with_name("necc_egg_prices_daily.csv")
REQUIRED_COLUMNS = {
    "date",
    "market",
    "category",
    "price",
    "price_filled",
}

app = FastAPI(title="NECC Egg Prices API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


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


df = load_prices()


def to_json_records(frame: pd.DataFrame) -> list[dict]:
    """Convert pandas NaN values to JSON null before FastAPI serializes rows."""
    cleaned = frame.astype(object).where(pd.notna(frame), None)
    return cleaned.to_dict(orient="records")


def filter_prices(
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
def root():
    return {
        "service": "NECC Egg Prices API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rows": len(df),
        "min_date": df["date"].min().date().isoformat(),
        "max_date": df["date"].max().date().isoformat(),
    }


@app.get("/prices/date-range")
def get_date_range():
    """Return the earliest and latest dates available in the dataset."""
    return {
        "min_date": df["date"].min().date().isoformat(),
        "max_date": df["date"].max().date().isoformat(),
    }

@app.get("/prices/markets")
def get_markets():
    """List all unique market names."""
    return {"markets": sorted(df["market"].unique().tolist())}

@app.get("/prices/categories")
def get_categories():
    """List all unique categories."""
    return {"categories": sorted(df["category"].unique().tolist())}

@app.get("/prices")
def get_prices(
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
    """
    Get egg price records with optional filters and pagination.
    """
    filtered = filter_prices(start_date, end_date, market, category, use_filled)
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
    start_date: Annotated[Optional[date], Query(description="YYYY-MM-DD")] = None,
    end_date: Annotated[Optional[date], Query(description="YYYY-MM-DD")] = None,
    market: Annotated[
        Optional[str],
        Query(description="Filter by market name (substring)"),
    ] = None,
    category: Annotated[Optional[str], Query(description="NECC or PREVAILING")] = None,
    use_filled: Annotated[bool, Query(description="Use interpolated prices")] = False,
):
    """
    Return average, min, max price per market and category within the date range.
    """
    filtered = filter_prices(start_date, end_date, market, category, use_filled)
    if filtered.empty:
        return {"stats": []}

    stats = (
        filtered.groupby(["market", "category"])["price"]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
        .round(2)
    )
    return {"stats": to_json_records(stats)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
