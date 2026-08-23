from pathlib import Path
import random
import time

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError


ROOT = Path(__file__).resolve().parents[1]

MASTER_FILE = ROOT / "data" / "processed" / "nse_mainboard_master_bse_classified.parquet"
STILL_UNMAPPED_FILE = ROOT / "data" / "processed" / "nse_bse_still_unmapped.csv"
YAHOO_MAPPING_FILE = ROOT / "data" / "processed" / "nse_yahoo_industry_mapping.csv"
YAHOO_UNMAPPED_FILE = ROOT / "data" / "processed" / "nse_yahoo_still_unmapped.csv"

BATCH_SIZE = 15
REQUEST_DELAY_SECONDS = 2.0
MAX_RETRIES = 5
BACKOFF_SECONDS = 30


def clean(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def fetch_yahoo_info(yahoo_ticker):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            info = yf.Ticker(yahoo_ticker).get_info() or {}
            return info, ""

        except YFRateLimitError:
            wait_seconds = (
                BACKOFF_SECONDS * (2 ** (attempt - 1))
                + random.uniform(0, 5)
            )

            print(
                f"  Yahoo rate limited. Retry {attempt}/{MAX_RETRIES} "
                f"after {wait_seconds:.1f} seconds."
            )

            time.sleep(wait_seconds)

        except Exception as exc:
            return {}, f"{type(exc).__name__}: {exc}"

    return {}, "Yahoo rate limit persisted after maximum retries"


def load_yahoo_mapping():
    columns = [
        "symbol",
        "company_name",
        "isin",
        "yahoo_ticker",
        "yahoo_sector",
        "yahoo_industry",
        "yahoo_quote_type",
        "classification_status",
        "classification_source",
        "yahoo_failure_reason",
    ]

    if not YAHOO_MAPPING_FILE.exists():
        return pd.DataFrame(columns=columns)

    mapping_df = pd.read_csv(
        YAHOO_MAPPING_FILE,
        dtype=str,
    ).fillna("")

    for column in columns:
        if column not in mapping_df.columns:
            mapping_df[column] = ""

    return mapping_df[columns]


def main():
    if not MASTER_FILE.exists():
        raise FileNotFoundError(f"Missing master file: {MASTER_FILE}")

    if not STILL_UNMAPPED_FILE.exists():
        raise FileNotFoundError(
            f"Missing BSE exception file: {STILL_UNMAPPED_FILE}"
        )

    master_df = pd.read_parquet(MASTER_FILE)

    bse_unmapped_df = pd.read_csv(
        STILL_UNMAPPED_FILE,
        dtype=str,
    ).fillna("")

    yahoo_columns = [
        "yahoo_ticker",
        "yahoo_sector",
        "yahoo_industry",
        "yahoo_quote_type",
        "yahoo_attempted",
        "yahoo_failure_reason",
    ]

    for column in yahoo_columns:
        if column not in master_df.columns:
            master_df[column] = ""

    bse_unmapped_isins = set(
        bse_unmapped_df["isin"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    candidates = master_df[
        master_df["classification_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("NOT_FOUND")
        & ~master_df["yahoo_attempted"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("YES")
        & master_df["isin"]
        .fillna("")
        .astype(str)
        .str.strip()
        .isin(bse_unmapped_isins)
    ].copy()

    candidates = candidates.head(BATCH_SIZE)

    print(f"Total BSE-unresolved securities: {len(bse_unmapped_df)}")
    print(f"Yahoo candidates in this batch: {len(candidates)}")

    if candidates.empty:
        final_unmapped = master_df[
            master_df["classification_status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("NOT_FOUND")
        ].copy()

        final_unmapped.to_csv(YAHOO_UNMAPPED_FILE, index=False)

        print("========== YAHOO FALLBACK COMPLETE ==========")
        print("No Yahoo fallback candidates remain.")
        print(f"Still unresolved after Yahoo: {len(final_unmapped)}")
        print(f"Saved: {YAHOO_UNMAPPED_FILE}")
        return

    yahoo_mapping_df = load_yahoo_mapping()
    new_mapping_rows = []

    for count, (master_index, row) in enumerate(candidates.iterrows(), start=1):
        symbol = clean(row.get("symbol"))
        company_name = clean(row.get("company_name"))
        isin = clean(row.get("isin"))
        yahoo_ticker = f"{symbol}.NS"

        print(
            f"[{count}/{len(candidates)}] "
            f"{symbol} | {yahoo_ticker} | {isin}"
        )

        info, failure_reason = fetch_yahoo_info(yahoo_ticker)

        yahoo_sector = clean(info.get("sector"))
        yahoo_industry = clean(info.get("industry"))
        yahoo_quote_type = clean(info.get("quoteType"))

        matched = bool(yahoo_sector or yahoo_industry)

        master_df.at[master_index, "yahoo_ticker"] = yahoo_ticker
        master_df.at[master_index, "yahoo_sector"] = yahoo_sector
        master_df.at[master_index, "yahoo_industry"] = yahoo_industry
        master_df.at[master_index, "yahoo_quote_type"] = yahoo_quote_type
        master_df.at[master_index, "yahoo_attempted"] = "YES"

        master_df.at[master_index, "yahoo_failure_reason"] = (
            "" if matched
            else failure_reason or "No Yahoo sector or industry found"
        )

        if matched:
            master_df.at[
                master_index,
                "classification_status",
            ] = "YAHOO_FALLBACK"

            master_df.at[
                master_index,
                "classification_source",
            ] = "Yahoo Finance"

            print(
                f"  RESOLVED | Sector: {yahoo_sector} | "
                f"Industry: {yahoo_industry}"
            )
        else:
            print(
                f"  Still unresolved: "
                f"{master_df.at[master_index, 'yahoo_failure_reason']}"
            )

        new_mapping_rows.append(
            {
                "symbol": symbol,
                "company_name": company_name,
                "isin": isin,
                "yahoo_ticker": yahoo_ticker,
                "yahoo_sector": yahoo_sector,
                "yahoo_industry": yahoo_industry,
                "yahoo_quote_type": yahoo_quote_type,
                "classification_status": (
                    "YAHOO_FALLBACK" if matched else "NOT_FOUND"
                ),
                "classification_source": (
                    "Yahoo Finance" if matched else ""
                ),
                "yahoo_failure_reason": (
                    "" if matched
                    else master_df.at[
                        master_index,
                        "yahoo_failure_reason",
                    ]
                ),
            }
        )

        time.sleep(REQUEST_DELAY_SECONDS)

    yahoo_mapping_df = pd.concat(
        [yahoo_mapping_df, pd.DataFrame(new_mapping_rows)],
        ignore_index=True,
    )

    yahoo_mapping_df = (
        yahoo_mapping_df
        .drop_duplicates(subset=["isin"], keep="last")
        .sort_values("symbol")
        .reset_index(drop=True)
    )

    master_df.to_parquet(MASTER_FILE, index=False)
    yahoo_mapping_df.to_csv(YAHOO_MAPPING_FILE, index=False)

    final_unmapped = master_df[
        master_df["classification_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("NOT_FOUND")
    ].copy()

    final_unmapped.to_csv(YAHOO_UNMAPPED_FILE, index=False)

    yahoo_resolved = int(
        master_df["classification_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("YAHOO_FALLBACK")
        .sum()
    )

    remaining_yahoo = int(
        (
            master_df["classification_status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("NOT_FOUND")
            & ~master_df["yahoo_attempted"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("YES")
            & master_df["isin"]
            .fillna("")
            .astype(str)
            .str.strip()
            .isin(bse_unmapped_isins)
        ).sum()
    )

    print("\n========== YAHOO FALLBACK BATCH COMPLETE ==========")
    print(f"Yahoo candidates processed this run: {len(candidates)}")
    print(f"Total Yahoo fallback classifications: {yahoo_resolved}")
    print(f"Still unresolved total: {len(final_unmapped)}")
    print(f"Yahoo candidates not yet attempted: {remaining_yahoo}")
    print(f"Updated master: {MASTER_FILE}")
    print(f"Yahoo mapping: {YAHOO_MAPPING_FILE}")
    print(f"Final unresolved report: {YAHOO_UNMAPPED_FILE}")


if __name__ == "__main__":
    main()
