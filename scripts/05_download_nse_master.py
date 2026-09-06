# scripts/05_download_nse_master.py
# Downloads the official NSE equity master for the classification pipeline.
# It deliberately does not classify securities; mapping is handled by scripts
# 06, 08 and 09. Existing classifications live in the classified master file.

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUT_DIR / "nse_mainboard_master.parquet"

NSE_HOME_URL = "https://www.nseindia.com"
NSE_EQUITY_MASTER_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
    "Accept-Language": "en-IN,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/securities-available-for-trading",
}

OUTPUT_COLUMNS = [
    "symbol",
    "company_name",
    "series",
    "isin",
    "listing_date",
    "face_value",
    "industry",
    "basic_industry",
    "sector",
    "classification_status",
    "classification_source",
    "classification_failure_reason",
    "master_downloaded_at_utc",
]


def clean_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def download_nse_csv() -> str:
    session = requests.Session()
    session.headers.update(HEADERS)
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            # NSE often expects a browser session/cookies before archive access.
            session.get(NSE_HOME_URL, timeout=20)
            response = session.get(NSE_EQUITY_MASTER_URL, timeout=45)
            response.raise_for_status()
            text = response.content.decode("utf-8-sig", errors="replace")
            header = text.splitlines()[0].upper() if text.splitlines() else ""
            if "SYMBOL" not in header or "ISIN" not in header:
                raise ValueError("NSE returned an unexpected response instead of EQUITY_L.csv")
            return text
        except Exception as exc:
            last_error = exc
            print(f"NSE master download attempt {attempt}/3 failed: {type(exc).__name__}: {exc}")
            if attempt < 3:
                time.sleep(attempt * 5)

    raise RuntimeError(f"Could not download NSE equity master: {last_error}")


def main() -> None:
    print("========== NSE MASTER DOWNLOAD START ==========")
    text = download_nse_csv()
    data = pd.read_csv(StringIO(text))
    data.columns = [str(column).strip() for column in data.columns]

    rename_map = {
        "SYMBOL": "symbol",
        "NAME OF COMPANY": "company_name",
        "SERIES": "series",
        "DATE OF LISTING": "listing_date",
        "ISIN NUMBER": "isin",
        "FACE VALUE": "face_value",
    }
    data = data.rename(columns=rename_map)

    required = ["symbol", "company_name", "series", "isin"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Unexpected NSE master columns. Missing: {missing}; found: {list(data.columns)}")

    for column in required:
        data[column] = clean_series(data[column])
    if "face_value" not in data.columns:
        data["face_value"] = ""
    data["face_value"] = clean_series(data["face_value"])
    data["listing_date"] = pd.to_datetime(data.get("listing_date"), errors="coerce", dayfirst=True)

    # Keep the official NSE master broader than the dashboard's EQ-only market
    # universe. The feature engine later applies its own strict EQ filter.
    data = data[data["series"].isin(["EQ", "BE", "BZ"])].copy()
    data = data[(data["symbol"] != "") & (data["isin"] != "")].copy()
    data = data.drop_duplicates(subset=["isin"], keep="last")

    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    data["industry"] = ""
    data["basic_industry"] = ""
    data["sector"] = ""
    data["classification_status"] = "PENDING"
    data["classification_source"] = ""
    data["classification_failure_reason"] = ""
    data["master_downloaded_at_utc"] = timestamp

    output = data[OUTPUT_COLUMNS].sort_values(["symbol", "series", "isin"]).reset_index(drop=True)
    output.to_parquet(OUTPUT_FILE, index=False)

    series_counts = output["series"].value_counts().to_dict()
    print("========== NSE MASTER DOWNLOAD COMPLETE ==========")
    print(f"Saved records: {len(output):,}")
    print(f"Series distribution: {series_counts}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Downloaded at: {timestamp}")


if __name__ == "__main__":
    main()
