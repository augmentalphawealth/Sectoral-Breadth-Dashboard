# scripts/11_update_nse_universe.py
# Daily NSE universe refresh only.
# This script preserves valid existing classifications and adds newly detected
# NSE EQ listings as PENDING. Dedicated BSE/Yahoo scripts classify them later.

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
import time
import urllib.error
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MASTER_FILE = PROCESSED / "nse_mainboard_master_bse_classified.parquet"

NSE_BHAVCOPY_BASE = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_"
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/live-equity-market",
}

MASTER_COLUMNS = [
    "symbol",
    "company_name",
    "series",
    "isin",
    "listing_date",
    "sector",
    "industry",
    "basic_industry",
    "classification_status",
    "classification_source",
    "classification_failure_reason",
    "yahoo_ticker",
    "yahoo_sector",
    "yahoo_industry",
    "yahoo_quote_type",
    "yahoo_attempted",
    "yahoo_failure_reason",
    "fallback_bse_attempted",
    "last_classification_attempt_utc",
]


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def clean_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def request_text(url: str, timeout: int = 45) -> str:
    request = urllib.request.Request(url, headers=NSE_HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8-sig", errors="replace")


def download_recent_bhavcopy() -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return the most recent available NSE bhavcopy within the last 10 days."""
    today_utc = datetime.now(timezone.utc).date()
    attempted: list[str] = []

    for offset in range(1, 11):
        candidate = today_utc - timedelta(days=offset)
        if candidate.weekday() >= 5:
            continue

        date_key = candidate.strftime("%d%m%Y")
        attempted.append(date_key)
        url = NSE_BHAVCOPY_BASE + f"sec_bhavdata_full_{date_key}.csv"

        try:
            text = request_text(url)
            frame = pd.read_csv(StringIO(text))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            print(f"Bhavcopy unavailable for {candidate.isoformat()}: {type(exc).__name__}")
            time.sleep(1)
            continue

        frame.columns = [str(column).strip() for column in frame.columns]
        rename_map = {
            "SYMBOL": "symbol",
            "SERIES": "series",
            "ISIN": "isin",
            "COMPANY NAME": "company_name",
        }
        frame = frame.rename(columns=rename_map)
        required = ["symbol", "series", "isin", "company_name"]
        if any(column not in frame.columns for column in required):
            print(f"Bhavcopy column mismatch for {candidate.isoformat()}: {list(frame.columns)}")
            continue

        for column in required:
            frame[column] = clean_series(frame[column])

        frame = frame[frame["series"].eq("EQ")].copy()
        frame = frame[(frame["symbol"] != "") & (frame["isin"] != "")].copy()
        frame["listing_date"] = pd.NaT
        frame = frame.drop_duplicates(subset=["isin"], keep="last")
        frame = frame.sort_values(["symbol", "series"]).reset_index(drop=True)

        if not frame.empty:
            return frame, pd.Timestamp(candidate)

    raise RuntimeError(
        "Could not download a usable NSE EQ bhavcopy. "
        f"Attempted dates: {', '.join(attempted)}"
    )


def ensure_master_schema(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()

    for column in MASTER_COLUMNS:
        if column not in data.columns:
            data[column] = ""

    for column in [
        "symbol",
        "company_name",
        "series",
        "isin",
        "sector",
        "industry",
        "basic_industry",
        "classification_status",
        "classification_source",
        "classification_failure_reason",
        "yahoo_ticker",
        "yahoo_sector",
        "yahoo_industry",
        "yahoo_quote_type",
        "yahoo_attempted",
        "yahoo_failure_reason",
        "fallback_bse_attempted",
        "last_classification_attempt_utc",
    ]:
        data[column] = clean_series(data[column])

    data["listing_date"] = pd.to_datetime(data["listing_date"], errors="coerce")
    return data


def is_complete_classification(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["sector"].ne("")
        & frame["industry"].ne("")
        & frame["basic_industry"].ne("")
    )


def main() -> None:
    print("========== NSE UNIVERSE UPDATE START ==========")

    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Missing classified master file: {MASTER_FILE}. "
            "Run scripts/05_download_nse_master.py and the mapping pipeline first."
        )

    current = ensure_master_schema(pd.read_parquet(MASTER_FILE))
    current = current[(current["isin"] != "") & (current["symbol"] != "")].copy()

    latest, bhavcopy_date = download_recent_bhavcopy()
    print(f"Bhavcopy date used: {bhavcopy_date.date()}")
    print(f"Existing classified-master rows: {len(current):,}")
    print(f"NSE EQ rows in bhavcopy: {len(latest):,}")

    current_isins = set(current["isin"])
    new_rows = latest[~latest["isin"].isin(current_isins)].copy()

    # Existing securities may have changed NSE trading symbol or company name.
    # Refresh only identity fields; retain all established mapping/classification.
    latest_lookup = latest.set_index("isin")
    matching_isins = current["isin"].isin(latest_lookup.index)
    for column in ["symbol", "company_name", "series"]:
        mapped = current.loc[matching_isins, "isin"].map(latest_lookup[column])
        current.loc[matching_isins & mapped.notna(), column] = mapped[mapped.notna()].astype(str)

    now_text = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if not new_rows.empty:
        for column in MASTER_COLUMNS:
            if column not in new_rows.columns:
                new_rows[column] = ""
        new_rows["listing_date"] = pd.to_datetime(new_rows["listing_date"], errors="coerce")
        new_rows["classification_status"] = "PENDING"
        new_rows["classification_source"] = ""
        new_rows["classification_failure_reason"] = "New NSE listing awaiting classification"
        new_rows["last_classification_attempt_utc"] = ""
        new_rows = new_rows[MASTER_COLUMNS]
        print(f"New NSE listings added as PENDING: {len(new_rows):,}")
    else:
        print("No new NSE EQ listings detected.")

    updated = pd.concat([current[MASTER_COLUMNS], new_rows], ignore_index=True)
    updated = updated.drop_duplicates(subset=["isin"], keep="last")
    updated = ensure_master_schema(updated)

    # Do not overwrite classifications. A row is only marked PENDING where all
    # three hierarchy fields are blank. Any incomplete older classification is
    # marked REVIEW_REQUIRED, allowing the downstream mapper to retry it.
    complete = is_complete_classification(updated)
    blank_status = updated["classification_status"].eq("")
    incomplete = ~complete
    updated.loc[blank_status & complete, "classification_status"] = "CLASSIFIED"
    updated.loc[blank_status & incomplete, "classification_status"] = "PENDING"
    updated.loc[incomplete & ~updated["classification_status"].eq("PENDING") & ~updated["classification_status"].eq("NOT_FOUND"), "classification_status"] = "REVIEW_REQUIRED"

    updated = updated.sort_values(["symbol", "series", "isin"]).reset_index(drop=True)
    MASTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    updated.to_parquet(MASTER_FILE, index=False)

    pending = int(updated["classification_status"].eq("PENDING").sum())
    review = int(updated["classification_status"].eq("REVIEW_REQUIRED").sum())
    complete_count = int(is_complete_classification(updated).sum())

    print("========== NSE UNIVERSE UPDATE COMPLETE ==========")
    print(f"Master rows: {len(updated):,}")
    print(f"Complete Sector/Industry/Basic Industry mappings: {complete_count:,}")
    print(f"Pending new mappings: {pending:,}")
    print(f"Incomplete mappings requiring review/retry: {review:,}")
    print(f"Updated: {MASTER_FILE}")
    print(f"Run timestamp: {now_text}")


if __name__ == "__main__":
    main()
