from pathlib import Path

import pandas as pd
import requests

from nse_tools import download_nse_mainboard_list
from bse_classification import classify_via_bse_api
from yahoo_classification import classify_via_yahoo_finance


ROOT = Path(__file__).resolve().parents[1]

MASTER_FILE = ROOT / "data" / "processed" / "nse_mainboard_master_bse_classified.parquet"
BSE_MAPPING_FILE = ROOT / "data" / "processed" / "nse_bse_industry_mapping.csv"
BSE_UNMAPPED_FILE = ROOT / "data" / "processed" / "nse_bse_still_unmapped.csv"
YAHOO_MAPPING_FILE = ROOT / "data" / "processed" / "nse_yahoo_industry_mapping.csv"
YAHOO_UNMAPPED_FILE = ROOT / "data" / "processed" / "nse_yahoo_still_unmapped.csv"


def clean(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def main():
    print("========== NSE UNIVERSE UPDATE START ==========")

    current = pd.read_parquet(MASTER_FILE)

    for column in [
        "classification_status",
        "classification_source",
        "sector",
        "industry",
        "basic_industry",
        "yahoo_ticker",
        "yahoo_sector",
        "yahoo_industry",
        "classification_failure_reason",
        "yahoo_failure_reason",
    ]:
        if column not in current.columns:
            current[column] = ""

    current["classification_status"] = (
        current["classification_status"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    print(f"Current master size: {len(current)}")

    try:
        latest = download_nse_mainboard_list()
    except Exception as error:
        print(f"Failed to download NSE list: {error}")
        return

    print(f"Latest NSE mainboard size: {len(latest)}")

    current_isins = set(current["isin"].dropna().astype(str).str.strip())

    latest["isin"] = (
        latest["isin"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    new_stocks = latest[
        ~latest["isin"].isin(current_isins)
    ].copy()

    if len(new_stocks) == 0:
        print("No new stocks detected. Universe unchanged.")
        return

    print(f"New stocks detected: {len(new_stocks)}")

    for column in [
        "classification_status",
        "classification_source",
        "sector",
        "industry",
        "basic_industry",
        "yahoo_ticker",
        "yahoo_sector",
        "yahoo_industry",
        "classification_failure_reason",
        "yahoo_failure_reason",
    ]:
        new_stocks[column] = ""

    new_stocks["classification_status"] = "PENDING"

    print("Classifying new stocks via BSE...")

    bse_results = []

    for index, row in new_stocks.iterrows():
        symbol = clean(row.get("symbol", ""))
        company_name = clean(row.get("company_name", ""))
        series = clean(row.get("series", ""))
        isin = clean(row["isin"])

        if not symbol or not isin:
            bse_results.append({
                "symbol": symbol,
                "isin": isin,
                "status": "FAILED",
                "reason": "Missing symbol or ISIN",
                "sector": "",
                "industry": "",
                "basic_industry": "",
            })
            continue

        result = classify_via_bse_api(symbol, company_name, series, isin)

        bse_results.append({
            "symbol": symbol,
            "isin": isin,
            "status": result.get("status", "FAILED"),
            "reason": result.get("reason", ""),
            "sector": result.get("sector", ""),
            "industry": result.get("industry", ""),
            "basic_industry": result.get("basic_industry", ""),
        })

    bse_df = pd.DataFrame(bse_results)

    new_stocks = new_stocks.merge(
        bse_df,
        on=["symbol", "isin"],
        how="left",
        suffixes=("", "_bse"),
    )

    classified_mask = new_stocks["status"] == "CLASSIFIED"

    new_stocks.loc[classified_mask, "classification_status"] = "CLASSIFIED"
    new_stocks.loc[classified_mask, "classification_source"] = "BSE"
    new_stocks.loc[classified_mask, "sector"] = new_stocks.loc[
        classified_mask, "sector"
    ]
    new_stocks.loc[classified_mask, "industry"] = new_stocks.loc[
        classified_mask, "industry"
    ]
    new_stocks.loc[classified_mask, "basic_industry"] = new_stocks.loc[
        classified_mask, "basic_industry"
    ]

    pending_mask = new_stocks["classification_status"] == "PENDING"

    print(f"BSE classified: {int(classified_mask.sum())}")
    print(f"Pending Yahoo fallback: {int(pending_mask.sum())}")

    if pending_mask.sum() > 0:
        print("Classifying pending stocks via Yahoo...")

        pending = new_stocks[pending_mask].copy()

        yahoo_results = []

        for index, row in pending.iterrows():
            symbol = clean(row.get("symbol", ""))
            company_name = clean(row.get("company_name", ""))
            series = clean(row.get("series", ""))
            isin = clean(row["isin"])

            if not symbol or not isin:
                yahoo_results.append({
                    "symbol": symbol,
                    "isin": isin,
                    "status": "FAILED",
                    "reason": "Missing symbol or ISIN",
                    "yahoo_ticker": "",
                    "yahoo_sector": "",
                    "yahoo_industry": "",
                })
                continue

            result = classify_via_yahoo_finance(
                symbol, company_name, series, isin
            )

            yahoo_results.append({
                "symbol": symbol,
                "isin": isin,
                "status": result.get("status", "FAILED"),
                "reason": result.get("reason", ""),
                "yahoo_ticker": result.get("yahoo_ticker", ""),
                "yahoo_sector": result.get("yahoo_sector", ""),
                "yahoo_industry": result.get("yahoo_industry", ""),
            })

        yahoo_df = pd.DataFrame(yahoo_results)

        pending = pending.merge(
            yahoo_df,
            on=["symbol", "isin"],
            how="left",
            suffixes=("", "_yahoo"),
        )

        yahoo_classified_mask = pending["status"] == "YAHOO_FALLBACK"

        pending.loc[
            yahoo_classified_mask, "classification_status"
        ] = "YAHOO_FALLBACK"
        pending.loc[
            yahoo_classified_mask, "classification_source"
        ] = "YAHOO"
        pending.loc[
            yahoo_classified_mask, "yahoo_ticker"
        ] = pending.loc[yahoo_classified_mask, "yahoo_ticker"]
        pending.loc[
            yahoo_classified_mask, "yahoo_sector"
        ] = pending.loc[yahoo_classified_mask, "yahoo_sector"]
        pending.loc[
            yahoo_classified_mask, "yahoo_industry"
        ] = pending.loc[yahoo_classified_mask, "yahoo_industry"]

        yahoo_failed_mask = pending["classification_status"] == "PENDING"

        pending.loc[
            yahoo_failed_mask, "classification_status"
        ] = "NOT_FOUND"
        pending.loc[
            yahoo_failed_mask, "classification_source"
        ] = ""

        new_stocks.loc[pending_mask] = pending

        print(f"Yahoo classified: {int(yahoo_classified_mask.sum())}")
        print(f"Still not found: {int(yahoo_failed_mask.sum())}")

    updated = pd.concat(
        [current, new_stocks],
        ignore_index=True,
    )

    updated = updated.sort_values(["symbol", "series"]).reset_index(drop=True)

    updated.to_parquet(MASTER_FILE, index=False)

    print("========== NSE UNIVERSE UPDATE COMPLETE ==========")
    print(f"Updated master size: {len(updated)}")
    print(f"New stocks added: {len(new_stocks)}")
    print(f"Updated master: {MASTER_FILE}")


if __name__ == "__main__":
    main()
