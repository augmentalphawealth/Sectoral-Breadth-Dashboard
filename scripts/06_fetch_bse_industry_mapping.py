from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import pandas as pd
from bse import BSE


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

MASTER_FILE = DATA_DIR / "nse_mainboard_master.parquet"
OUTPUT_FILE = DATA_DIR / "bse_industry_mapping.parquet"
UNMATCHED_FILE = DATA_DIR / "bse_industry_unmatched.csv"

SLEEP_SECONDS = 0.7


def first_value(value):
    """Return first usable text value from BSE lookup response."""
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (list, tuple)):
        for item in value:
            text = first_value(item)
            if text:
                return text
        return ""

    if isinstance(value, dict):
        for key in (
            "industry",
            "Industry",
            "sector",
            "Sector",
            "name",
            "Name",
            "company_name",
            "CompanyName",
            "scrip_name",
            "ScripName",
        ):
            if key in value and value[key]:
                text = first_value(value[key])
                if text:
                    return text
        return ""

    return str(value).strip()


def lookup_bse(bse: BSE, isin: str, symbol: str) -> dict:
    """Try ISIN first; symbol only as a fallback."""
    result = None
    lookup_method = getattr(bse, "lookup", None)

    if lookup_method is None:
        raise RuntimeError("Installed BSE package does not provide lookup().")

    try:
        result = lookup_method(isin)
    except Exception:
        result = None

    if not result:
        try:
            result = lookup_method(symbol)
        except Exception:
            result = None

    if not result:
        return {
            "bse_lookup_status": "UNMATCHED",
            "bse_raw": "",
            "bse_industry": "",
            "bse_sector": "",
        }

    industry = ""
    sector = ""

    if isinstance(result, dict):
        industry = first_value(
            result.get("industry")
            or result.get("Industry")
            or result.get("industry_name")
            or result.get("IndustryName")
        )
        sector = first_value(
            result.get("sector")
            or result.get("Sector")
            or result.get("sector_name")
            or result.get("SectorName")
        )

    return {
        "bse_lookup_status": "MATCHED",
        "bse_raw": str(result),
        "bse_industry": industry,
        "bse_sector": sector,
    }


def main():
    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"Missing {MASTER_FILE}. Run NSE Mainboard Master Update first."
        )

    master = pd.read_parquet(MASTER_FILE)
    master["isin"] = master["isin"].fillna("").astype(str).str.strip()
    master["symbol"] = master["symbol"].fillna("").astype(str).str.strip()

    if OUTPUT_FILE.exists():
        old = pd.read_parquet(OUTPUT_FILE)
        completed_isins = set(old["isin"].fillna("").astype(str))
    else:
        old = pd.DataFrame()
        completed_isins = set()

    pending = master[
        (master["isin"] != "") & (~master["isin"].isin(completed_isins))
    ].copy()

    print(f"Total NSE records: {len(master):,}")
    print(f"Already cached: {len(completed_isins):,}")
    print(f"To query: {len(pending):,}")

    bse = BSE()
    rows = []

    try:
        for number, row in enumerate(pending.itertuples(index=False), start=1):
            isin = row.isin
            symbol = row.symbol

            try:
                info = lookup_bse(bse, isin, symbol)
            except Exception as exc:
                info = {
                    "bse_lookup_status": "ERROR",
                    "bse_raw": "",
                    "bse_industry": "",
                    "bse_sector": "",
                    "error_message": str(exc),
                }

            rows.append(
                {
                    "isin": isin,
                    "symbol": symbol,
                    "company_name": getattr(row, "company_name", ""),
                    "bse_industry": info.get("bse_industry", ""),
                    "bse_sector": info.get("bse_sector", ""),
                    "bse_lookup_status": info.get("bse_lookup_status", ""),
                    "bse_raw": info.get("bse_raw", ""),
                    "error_message": info.get("error_message", ""),
                    "fetched_at": datetime.now().isoformat(timespec="seconds"),
                }
            )

            if number % 25 == 0 or number == len(pending):
                print(f"Processed {number:,}/{len(pending):,}")

            time.sleep(SLEEP_SECONDS)

    finally:
        try:
            bse.exit()
        except Exception:
            pass

    new = pd.DataFrame(rows)

    if len(old):
        combined = pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["isin"], keep="last")
    else:
        combined = new

    combined.to_parquet(OUTPUT_FILE, index=False)

    unmatched = combined[
        combined["bse_lookup_status"].isin(["UNMATCHED", "ERROR"])
    ].copy()
    unmatched.to_csv(UNMATCHED_FILE, index=False)

    print(f"Saved mapping cache: {OUTPUT_FILE}")
    print(f"Saved unmatched report: {UNMATCHED_FILE}")
    print(
        f"Matched: {(combined['bse_lookup_status'] == 'MATCHED').sum():,} | "
        f"Unmatched/Error: {len(unmatched):,}"
    )


if __name__ == "__main__":
    main()
