import argparse
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.e2necc.com/home/eggprice"
OUTPUT_FILE = "necc_egg_prices_daily.csv"
RAW_DIR = Path("raw_html")

DEFAULT_START_DATE = date(2009, 1, 1)
DEFAULT_END_DATE = date.today()
REQUEST_DELAY_SECONDS = 1
DEFAULT_INTERPOLATION_LIMIT_DAYS = 15

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/148.0.0.0 Safari/537.36"
    )
}


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def iter_months(start_date: date, end_date: date):
    year = start_date.year
    month = start_date.month

    while (year, month) <= (end_date.year, end_date.month):
        yield year, month

        if month == 12:
            year += 1
            month = 1
        else:
            month += 1


def download_month(year: int, month: int, session: requests.Session) -> str:
    payload = {
        "ddlMonth": f"{month:02d}",
        "ddlYear": str(year),
        "rblReportType": "DailyReport",
        "btnReport": "Get Sheet",
    }

    response = session.post(
        BASE_URL,
        data=payload,
        headers=HEADERS,
        timeout=60,
    )
    response.raise_for_status()
    return response.text


def read_or_download_month(
    year: int,
    month: int,
    session: requests.Session,
    raw_dir: Path,
    use_cache: bool,
) -> str:
    raw_file = raw_dir / f"egg_{year}_{month:02d}.html"

    if use_cache and raw_file.exists():
        print(f"Reading cached {year}-{month:02d}")
        return raw_file.read_text(encoding="utf-8")

    print(f"Downloading {year}-{month:02d}")
    html = download_month(year, month, session)
    raw_file.write_text(html, encoding="utf-8")
    time.sleep(REQUEST_DELAY_SECONDS)
    return html


def clean_price(value: str):
    value = value.strip().replace(",", "")

    if value in {"", "-", "—", "NA", "N/A"}:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def parse_daily_records(
    html: str,
    year: int,
    month: int,
    start_date: date,
    end_date: date,
):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", attrs={"border": "1px"})

    if table is None:
        print(f"Table not found for {year}-{month:02d}")
        return []

    current_section = None
    records = []

    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])

        if not cells:
            continue

        first_cell_text = cells[0].get_text(" ", strip=True)
        first_cell_upper = first_cell_text.upper()

        if "NECC SUGGESTED EGG PRICES" in first_cell_upper:
            current_section = "NECC"
            continue

        if "PREVAILING PRICES" in first_cell_upper:
            current_section = "PREVAILING"
            continue

        if first_cell_upper.startswith("NAME OF ZONE"):
            continue

        if current_section is None or len(cells) < 5:
            continue

        market = first_cell_text
        day_cells = cells[1:-1]

        for day, cell in enumerate(day_cells, start=1):
            try:
                record_date = date(year, month, day)
            except ValueError:
                continue

            if record_date < start_date or record_date > end_date:
                continue

            records.append(
                {
                    "date": record_date.isoformat(),
                    "year": record_date.year,
                    "month": record_date.month,
                    "day": record_date.day,
                    "market": market,
                    "category": current_section,
                    "price": clean_price(cell.get_text(strip=True)),
                }
            )

    return records


def complete_daily_grid(
    df: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    columns = ["date", "year", "month", "day", "market", "category", "price"]

    if df.empty:
        return pd.DataFrame(columns=columns)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    market_categories = (
        df[["market", "category"]]
        .drop_duplicates()
        .sort_values(["category", "market"])
    )
    dates = pd.date_range(start_date, end_date, freq="D")

    full_index = pd.DataFrame({"date": dates}).merge(
        market_categories,
        how="cross",
    )

    df = df[["date", "market", "category", "price"]]
    df = full_index.merge(
        df,
        on=["date", "market", "category"],
        how="left",
    )

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["date"] = df["date"].dt.date.astype(str)

    return df[columns]


def add_fill_columns(
    df: pd.DataFrame,
    interpolation_limit_days: int = DEFAULT_INTERPOLATION_LIMIT_DAYS,
) -> pd.DataFrame:
    columns = [
        "date",
        "year",
        "month",
        "day",
        "market",
        "category",
        "price",
        "price_filled",
        "fill_method",
    ]

    if df.empty:
        return pd.DataFrame(columns=columns)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["market", "category", "date"], inplace=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price_filled"] = df["price"]
    df["fill_method"] = "observed"

    original_missing = df["price"].isna()
    df.loc[original_missing, "fill_method"] = "missing"

    group_keys = ["market", "category"]

    interpolated = df["price_filled"].copy()

    for _, group in df.groupby(group_keys, sort=False):
        prices = group["price_filled"]
        linear_prices = prices.interpolate(method="linear")
        missing = prices.isna()
        run_ids = missing.ne(missing.shift()).cumsum()

        for _, run in group[missing].groupby(run_ids[missing]):
            positions = group.index.get_indexer(run.index)
            start_position = positions[0]
            end_position = positions[-1]
            gap_days = len(run)
            has_previous_price = (
                start_position > 0
                and pd.notna(prices.iloc[start_position - 1])
            )
            has_next_price = (
                end_position < len(prices) - 1
                and pd.notna(prices.iloc[end_position + 1])
            )

            if (
                gap_days <= interpolation_limit_days
                and has_previous_price
                and has_next_price
            ):
                interpolated.loc[run.index] = linear_prices.loc[run.index]

    interpolated_mask = original_missing & interpolated.notna()
    df.loc[interpolated_mask, "price_filled"] = interpolated[interpolated_mask]
    df.loc[interpolated_mask, "fill_method"] = "linear_interpolation"

    previous_filled = df.groupby(group_keys)["price_filled"].ffill()
    previous_mask = original_missing & df["price_filled"].isna() & previous_filled.notna()
    df.loc[previous_mask, "price_filled"] = previous_filled[previous_mask]
    df.loc[previous_mask, "fill_method"] = "previous_price"

    next_filled = df.groupby(group_keys)["price_filled"].bfill()
    next_mask = original_missing & df["price_filled"].isna() & next_filled.notna()
    df.loc[next_mask, "price_filled"] = next_filled[next_mask]
    df.loc[next_mask, "fill_method"] = "next_price"

    df.loc[df["price_filled"].isna(), "fill_method"] = "no_source_data"
    df["date"] = df["date"].dt.date.astype(str)
    df.sort_values(["date", "category", "market"], inplace=True)

    return df[columns]


def build_daily_dataset(
    start_date: date,
    end_date: date,
    raw_dir: Path,
    use_cache: bool,
) -> pd.DataFrame:
    raw_dir.mkdir(exist_ok=True)
    session = requests.Session()
    all_records = []

    for year, month in iter_months(start_date, end_date):
        try:
            html = read_or_download_month(
                year=year,
                month=month,
                session=session,
                raw_dir=raw_dir,
                use_cache=use_cache,
            )
            records = parse_daily_records(
                html=html,
                year=year,
                month=month,
                start_date=start_date,
                end_date=end_date,
            )
            print(f"  -> {len(records)} daily rows")
            all_records.extend(records)
        except Exception as exc:
            print(f"FAILED {year}-{month:02d}: {exc}")

    columns = ["date", "year", "month", "day", "market", "category", "price"]
    df = pd.DataFrame(all_records, columns=columns)

    if not df.empty:
        df.drop_duplicates(
            subset=["date", "market", "category"],
            keep="last",
            inplace=True,
        )
        df.sort_values(["date", "category", "market"], inplace=True)
        df = complete_daily_grid(df, start_date, end_date)
        df.sort_values(["date", "category", "market"], inplace=True)

    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download NECC egg prices and save daily data."
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=DEFAULT_START_DATE,
        help="First daily date to include, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=DEFAULT_END_DATE,
        help="Last daily date to include, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help="CSV file path for the daily dataset.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory used to store downloaded monthly HTML pages.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Download pages again even if raw HTML already exists.",
    )
    parser.add_argument(
        "--interpolation-limit-days",
        type=int,
        default=DEFAULT_INTERPOLATION_LIMIT_DAYS,
        help="Maximum consecutive missing days to fill with linear interpolation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.end_date < args.start_date:
        raise ValueError("end date must be on or after start date")

    df = build_daily_dataset(
        start_date=args.start_date,
        end_date=args.end_date,
        raw_dir=args.raw_dir,
        use_cache=not args.no_cache,
    )
    df = add_fill_columns(
        df,
        interpolation_limit_days=args.interpolation_limit_days,
    )
    df.to_csv(args.output, index=False, na_rep="NA")

    print()
    print("Finished")
    print(f"Daily date range: {args.start_date} to {args.end_date}")
    print(f"Rows: {len(df):,}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()