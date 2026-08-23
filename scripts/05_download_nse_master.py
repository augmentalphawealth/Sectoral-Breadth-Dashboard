from __future__ import annotations

from io import StringIO
from pathlib import Path
import time

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NSE_HOME_URL = "https://www.nseindia.com"
NSE_EQUITY_MASTER_URL = (
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
)

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


def download_nse_csv() -> str:
    session = requests.Session()
    session.headers.update(HEADERS)

    for attempt in range(1, 4):
        try:
            session.get(NSE_HOME_URL, timeout=20)
            response = session.get(NSE_EQUITY_MASTER_URL, timeout=45)
            response.raise_for_status()

            text = response.content.decode("utf-8-sig", errors="replace")
            if "SYMBOL" not in text.upper() or "ISIN" not in text.upper():
                raise ValueError(
                    "NSE returned an unexpected response instead of EQUITY_L.csv"
                )
            return text

        except Exception as exc:
            print(f"Download attempt {attempt}/3 failed: {exc}")
            if attempt == 3:
                raise
            time.sleep(attempt * 5)

    raise RuntimeError("Could not download NSE equity master")


def main():
    print("Downloading NSE equity master...")

    csv_text = download_nse_csv()
    df = pd.read_csv(StringIO(csv_text))
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "SYMBOL": "symbol",
        "NAME OF COMPANY": "company_name",
        "SERIES": "series",
        "DATE OF LISTING": "listing_date",
        "ISIN NUMBER": "isin",
        "FACE VALUE": "face_value",
    }
    df = df.rename(columns=rename_map)

    required = ["symbol", "company_name", "series", "isin"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Unexpected NSE file structure. Missing columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["company_name"] = df["company_name"].astype(str).str.strip()
    df["series"] = df["series"].astype(str).str.strip()
    df["isin"] = df["isin"].astype(str).str.strip()

    # Include normal EQ shares plus BE/BZ series for your fallback-aware NSE universe.
    df = df[df["series"].isin(["EQ", "BE", "BZ"])].copy()

    df["listing_date"] = pd.to_datetime(
        df.get("listing_date"), errors="coerce", dayfirst=True
    )

    df["classification_status"] = "PENDING"
    df["classification_source"] = ""
    df["industry"] = ""
    df["basic_industry"] = ""
    df["sector"] = ""

    keep = [
        "symbol",
        "company_name",
        "series",
        "isin",
        "listing_date",
        "industry",
        "basic_industry",
        "sector",
        "classification_status",
        "classification_source",
    ]

    out = df[keep].drop_duplicates(subset=["isin"], keep="last")
    out = out.sort_values("symbol").reset_index(drop=True)

    output_file = OUT_DIR / "nse_mainboard_master.parquet"
    out.to_parquet(output_file, index=False)

    print(f"Saved {len(out):,} NSE company records")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
