# scripts/07_export_bse_unmapped.py
# Exports all incomplete/retryable classifications for BSE fallback processing.

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
INPUT_FILE = PROCESSED / "nse_mainboard_master_bse_classified.parquet"
OUTPUT_FILE = PROCESSED / "nse_bse_unmapped.csv"

RETRYABLE_STATUSES = {"PENDING", "REVIEW_REQUIRED", "NOT_FOUND", "BSE_RETRY"}

REPORT_COLUMNS = [
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
    "bse_attempt_count",
    "bse_last_attempt_utc",
    "bse_code",
    "fallback_bse_attempted",
    "yahoo_attempted",
]


def clean_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing classified master file: {INPUT_FILE}")

    master = pd.read_parquet(INPUT_FILE).copy()
    required = ["symbol", "isin", "classification_status", "sector", "industry", "basic_industry"]
    missing = [column for column in required if column not in master.columns]
    if missing:
        raise ValueError(f"Master file missing required columns: {missing}")

    for column in [
        "symbol", "company_name", "series", "isin", "classification_status",
        "classification_source", "sector", "industry", "basic_industry",
        "classification_failure_reason", "bse_attempt_count", "bse_last_attempt_utc",
        "bse_code", "fallback_bse_attempted", "yahoo_attempted",
    ]:
        if column not in master.columns:
            master[column] = ""
        master[column] = clean_series(master[column])

    hierarchy_complete = (
        master["sector"].ne("")
        & master["industry"].ne("")
        & master["basic_industry"].ne("")
    )
    retryable = master["classification_status"].isin(RETRYABLE_STATUSES)
    unmapped = master[~hierarchy_complete & retryable].copy()

    # Avoid blank identity records: they cannot safely be resolved by BSE/Yahoo.
    unmapped = unmapped[(unmapped["symbol"] != "") & (unmapped["isin"] != "")].copy()
    unmapped["report_generated_at_utc"] = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )

    columns = [column for column in REPORT_COLUMNS if column in unmapped.columns]
    columns.append("report_generated_at_utc")
    unmapped = unmapped[columns].drop_duplicates(subset=["isin"], keep="last")
    unmapped = unmapped.sort_values(["classification_status", "bse_attempt_count", "symbol", "isin"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    unmapped.to_csv(OUTPUT_FILE, index=False)

    status_counts = master["classification_status"].value_counts(dropna=False).to_dict()
    print("========== CLASSIFICATION EXCEPTION REPORT COMPLETE ==========")
    print(f"Total master records: {len(master):,}")
    print(f"Complete hierarchy records: {int(hierarchy_complete.sum()):,}")
    print(f"Retryable incomplete records exported: {len(unmapped):,}")
    print(f"Status counts: {status_counts}")
    print(f"Saved report: {OUTPUT_FILE}")

    if not unmapped.empty:
        print("\n========== FIRST 25 RETRYABLE RECORDS ==========")
        print(unmapped.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
