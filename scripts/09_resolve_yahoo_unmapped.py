# scripts/09_resolve_yahoo_unmapped.py
# Yahoo Finance fallback for records still incomplete after BSE attempts.
# Yahoo fields are retained as source data/audit data. This script does not
# invent a Basic Industry from Yahoo's different classification taxonomy.

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import random
import time

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MASTER_FILE = PROCESSED / "nse_mainboard_master_bse_classified.parquet"
STILL_UNMAPPED_FILE = PROCESSED / "nse_bse_still_unmapped.csv"
YAHOO_MAPPING_FILE = PROCESSED / "nse_yahoo_industry_mapping.csv"
YAHOO_UNMAPPED_FILE = PROCESSED / "nse_yahoo_still_unmapped.csv"

BATCH_SIZE = 15
REQUEST_DELAY_SECONDS = 2.0
MAX_RETRIES = 4
BACKOFF_SECONDS = 20
RETRYABLE_STATUSES = {"PENDING", "REVIEW_REQUIRED", "NOT_FOUND", "BSE_RETRY", "YAHOO_RETRY"}

YAHOO_MAPPING_COLUMNS = [
    "isin", "symbol", "company_name", "yahoo_ticker", "yahoo_sector",
    "yahoo_industry", "yahoo_quote_type", "classification_status",
    "classification_source", "yahoo_failure_reason", "attempted_at_utc",
    "attempt_count",
]


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in [
        "symbol", "company_name", "series", "isin", "sector", "industry",
        "basic_industry", "classification_status", "classification_source",
        "classification_failure_reason", "yahoo_ticker", "yahoo_sector",
        "yahoo_industry", "yahoo_quote_type", "yahoo_attempt_count",
        "yahoo_last_attempt_utc", "yahoo_attempted", "yahoo_failure_reason",
    ]:
        if column not in data.columns:
            data[column] = ""
        data[column] = text_series(data[column])
    data["yahoo_attempt_count"] = pd.to_numeric(data["yahoo_attempt_count"], errors="coerce").fillna(0).astype(int)
    return data


def complete_hierarchy(frame: pd.DataFrame) -> pd.Series:
    return frame["sector"].ne("") & frame["industry"].ne("") & frame["basic_industry"].ne("")


def fetch_yahoo_info(ticker: str) -> tuple[dict, str]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return yf.Ticker(ticker).get_info() or {}, ""
        except YFRateLimitError:
            wait = BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 3)
            print(f"  Yahoo rate-limited; retry {attempt}/{MAX_RETRIES} in {wait:.1f} seconds")
            time.sleep(wait)
        except Exception as exc:
            return {}, f"{type(exc).__name__}: {exc}"
    return {}, "Yahoo rate limit persisted after maximum retries"


def load_mapping() -> pd.DataFrame:
    if not YAHOO_MAPPING_FILE.exists():
        return pd.DataFrame(columns=YAHOO_MAPPING_COLUMNS)
    mapping = pd.read_csv(YAHOO_MAPPING_FILE, dtype=str).fillna("")
    for column in YAHOO_MAPPING_COLUMNS:
        if column not in mapping.columns:
            mapping[column] = ""
    mapping["attempt_count"] = pd.to_numeric(mapping["attempt_count"], errors="coerce").fillna(0).astype(int)
    return mapping[YAHOO_MAPPING_COLUMNS]


def write_final_unmapped(master: pd.DataFrame) -> pd.DataFrame:
    unresolved = master[~complete_hierarchy(master)].copy()
    unresolved = unresolved[text_series(unresolved["classification_status"]).isin(RETRYABLE_STATUSES | {"YAHOO_RETRY"})].copy()
    columns = [
        "symbol", "company_name", "series", "isin", "listing_date", "sector",
        "industry", "basic_industry", "classification_status", "classification_source",
        "classification_failure_reason", "yahoo_ticker", "yahoo_sector",
        "yahoo_industry", "yahoo_quote_type", "yahoo_attempt_count",
        "yahoo_last_attempt_utc", "yahoo_failure_reason",
    ]
    columns = [column for column in columns if column in unresolved.columns]
    unresolved = unresolved[columns].drop_duplicates("isin", keep="last").sort_values(["symbol", "isin"])
    unresolved.to_csv(YAHOO_UNMAPPED_FILE, index=False)
    return unresolved


def main() -> None:
    print("========== YAHOO FALLBACK START ==========")
    if not MASTER_FILE.exists():
        raise FileNotFoundError(f"Missing master file: {MASTER_FILE}")
    if not STILL_UNMAPPED_FILE.exists():
        raise FileNotFoundError(f"Missing BSE exception report: {STILL_UNMAPPED_FILE}. Run script 08 first.")

    master = ensure_columns(pd.read_parquet(MASTER_FILE))
    bse_unmapped = pd.read_csv(STILL_UNMAPPED_FILE, dtype=str).fillna("")
    if "isin" not in bse_unmapped.columns:
        raise ValueError(f"BSE exception report has no ISIN column: {STILL_UNMAPPED_FILE}")
    bse_isins = set(text_series(bse_unmapped["isin"])) - {""}

    candidates = master[
        master["isin"].isin(bse_isins)
        & ~complete_hierarchy(master)
        & text_series(master["classification_status"]).isin(RETRYABLE_STATUSES)
        & master["symbol"].ne("")
    ].copy()
    candidates = candidates.sort_values(["yahoo_attempt_count", "symbol", "isin"]).head(BATCH_SIZE)

    print(f"BSE-incomplete report rows: {len(bse_unmapped):,}")
    print(f"Yahoo candidates this run: {len(candidates):,}")

    if candidates.empty:
        unresolved = write_final_unmapped(master)
        master.to_parquet(MASTER_FILE, index=False)
        print(f"No eligible Yahoo candidates. Still incomplete: {len(unresolved):,}")
        return

    mapping = load_mapping()
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    records: list[dict] = []

    for number, (index, row) in enumerate(candidates.iterrows(), start=1):
        symbol = clean(row["symbol"])
        isin = clean(row["isin"])
        ticker = f"{symbol}.NS"
        prior_attempts = int(master.at[index, "yahoo_attempt_count"])
        print(f"[{number}/{len(candidates)}] {symbol} | {ticker} | prior attempts: {prior_attempts}")

        info, failure_reason = fetch_yahoo_info(ticker)
        yahoo_sector = clean(info.get("sector"))
        yahoo_industry = clean(info.get("industry"))
        yahoo_quote_type = clean(info.get("quoteType"))
        has_yahoo_profile = bool(yahoo_sector or yahoo_industry)

        master.at[index, "yahoo_ticker"] = ticker
        master.at[index, "yahoo_sector"] = yahoo_sector
        master.at[index, "yahoo_industry"] = yahoo_industry
        master.at[index, "yahoo_quote_type"] = yahoo_quote_type
        master.at[index, "yahoo_attempt_count"] = prior_attempts + 1
        master.at[index, "yahoo_last_attempt_utc"] = timestamp
        master.at[index, "yahoo_attempted"] = "YES"

        if has_yahoo_profile:
            # Yahoo is not assumed to provide a compatible Basic Industry
            # hierarchy. Preserve BSE/NSE taxonomy as blank/retryable until it
            # is genuinely available, while retaining Yahoo facts for review.
            master.at[index, "classification_status"] = "YAHOO_RETRY"
            master.at[index, "classification_source"] = "Yahoo Finance profile (hierarchy incomplete)"
            master.at[index, "classification_failure_reason"] = "Yahoo provided sector/industry but no verified Basic Industry mapping"
            master.at[index, "yahoo_failure_reason"] = ""
            status = "YAHOO_RETRY"
            reason = master.at[index, "classification_failure_reason"]
            print(f"  Yahoo profile saved | Sector: {yahoo_sector or '—'} | Industry: {yahoo_industry or '—'}")
        else:
            master.at[index, "classification_status"] = "YAHOO_RETRY"
            master.at[index, "classification_failure_reason"] = failure_reason or "Yahoo returned no sector or industry"
            master.at[index, "yahoo_failure_reason"] = master.at[index, "classification_failure_reason"]
            status = "YAHOO_RETRY"
            reason = master.at[index, "yahoo_failure_reason"]
            print(f"  Still retryable: {reason}")

        records.append({
            "isin": isin,
            "symbol": symbol,
            "company_name": clean(row["company_name"]),
            "yahoo_ticker": ticker,
            "yahoo_sector": yahoo_sector,
            "yahoo_industry": yahoo_industry,
            "yahoo_quote_type": yahoo_quote_type,
            "classification_status": status,
            "classification_source": clean(master.at[index, "classification_source"]),
            "yahoo_failure_reason": reason,
            "attempted_at_utc": timestamp,
            "attempt_count": prior_attempts + 1,
        })
        time.sleep(REQUEST_DELAY_SECONDS)

    mapping = pd.concat([mapping, pd.DataFrame(records)], ignore_index=True)
    mapping["attempt_count"] = pd.to_numeric(mapping["attempt_count"], errors="coerce").fillna(0).astype(int)
    mapping = mapping.drop_duplicates("isin", keep="last").sort_values(["symbol", "isin"]).reset_index(drop=True)
    master = master.drop_duplicates("isin", keep="last").sort_values(["symbol", "series", "isin"]).reset_index(drop=True)
    master.to_parquet(MASTER_FILE, index=False)
    mapping.to_csv(YAHOO_MAPPING_FILE, index=False)
    unresolved = write_final_unmapped(master)

    print("========== YAHOO FALLBACK COMPLETE ==========")
    print(f"Candidates processed this run: {len(records):,}")
    print(f"Complete Sector/Industry/Basic Industry mappings: {int(complete_hierarchy(master).sum()):,}")
    print(f"Still incomplete/retryable: {len(unresolved):,}")
    print(f"Updated master: {MASTER_FILE}")
    print(f"Yahoo audit: {YAHOO_MAPPING_FILE}")
    print(f"Unresolved report: {YAHOO_UNMAPPED_FILE}")


if __name__ == "__main__":
    main()
