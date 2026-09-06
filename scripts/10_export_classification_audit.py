# scripts/10_export_classification_audit.py
# Daily classification coverage and exception audit.

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MASTER_FILE = PROCESSED / "nse_mainboard_master_bse_classified.parquet"
SUMMARY_FILE = PROCESSED / "nse_classification_summary.csv"
REVIEW_FILE = PROCESSED / "nse_classification_review.csv"


REQUIRED_COLUMNS = [
    "symbol",
    "company_name",
    "series",
    "isin",
    "listing_date",
    "classification_status",
    "classification_source",
    "sector",
    "industry",
    "basic_industry",
    "classification_failure_reason",
    "bse_attempt_count",
    "bse_last_attempt_utc",
    "fallback_bse_attempt_count",
    "fallback_bse_last_attempt_utc",
    "yahoo_ticker",
    "yahoo_sector",
    "yahoo_industry",
    "yahoo_quote_type",
    "yahoo_attempt_count",
    "yahoo_last_attempt_utc",
    "yahoo_failure_reason",
]


def clean_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def main() -> None:
    if not MASTER_FILE.exists():
        raise FileNotFoundError(f"Missing classified master file: {MASTER_FILE}")

    data = pd.read_parquet(MASTER_FILE).copy()
    for column in REQUIRED_COLUMNS:
        if column not in data.columns:
            data[column] = ""

    for column in REQUIRED_COLUMNS:
        if column != "listing_date":
            data[column] = clean_series(data[column])
    data["listing_date"] = pd.to_datetime(data["listing_date"], errors="coerce")

    complete_hierarchy = (
        data["sector"].ne("")
        & data["industry"].ne("")
        & data["basic_industry"].ne("")
    )
    data["hierarchy_complete"] = complete_hierarchy
    data["classification_coverage"] = data[["sector", "industry", "basic_industry"]].ne("").sum(axis=1)

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    total = len(data)
    complete_count = int(complete_hierarchy.sum())
    incomplete_count = total - complete_count
    coverage_percent = (complete_count / total * 100.0) if total else 0.0

    status_summary = (
        data.groupby(["classification_status", "classification_source"], dropna=False)
        .size()
        .reset_index(name="stock_count")
    )
    status_summary["report_type"] = "status_source"

    coverage_summary = pd.DataFrame(
        [
            {"report_type": "coverage", "metric": "total_records", "value": total},
            {"report_type": "coverage", "metric": "complete_hierarchy_records", "value": complete_count},
            {"report_type": "coverage", "metric": "incomplete_hierarchy_records", "value": incomplete_count},
            {"report_type": "coverage", "metric": "complete_hierarchy_percent", "value": round(coverage_percent, 1)},
        ]
    )

    status_summary["metric"] = ""
    status_summary["value"] = ""
    output_columns = ["report_type", "classification_status", "classification_source", "stock_count", "metric", "value"]
    for frame in [status_summary, coverage_summary]:
        for column in output_columns:
            if column not in frame.columns:
                frame[column] = ""
    summary = pd.concat([coverage_summary[output_columns], status_summary[output_columns]], ignore_index=True)
    summary.insert(0, "generated_at_utc", generated_at)
    summary = summary.sort_values(["report_type", "classification_status", "classification_source", "metric"], na_position="last")
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_FILE, index=False)

    review = data[~complete_hierarchy].copy()
    review_columns = [
        "symbol", "company_name", "series", "isin", "listing_date",
        "classification_status", "classification_source", "classification_coverage",
        "sector", "industry", "basic_industry", "classification_failure_reason",
        "bse_attempt_count", "bse_last_attempt_utc", "fallback_bse_attempt_count",
        "fallback_bse_last_attempt_utc", "yahoo_ticker", "yahoo_sector",
        "yahoo_industry", "yahoo_quote_type", "yahoo_attempt_count",
        "yahoo_last_attempt_utc", "yahoo_failure_reason",
    ]
    review = review[review_columns].drop_duplicates(subset=["isin"], keep="last")
    review.insert(0, "report_generated_at_utc", generated_at)
    review = review.sort_values(
        ["classification_coverage", "classification_status", "symbol", "isin"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    review.to_csv(REVIEW_FILE, index=False)

    print("========== CLASSIFICATION AUDIT COMPLETE ==========")
    print(f"Total master records: {total:,}")
    print(f"Complete Sector/Industry/Basic Industry mappings: {complete_count:,} ({coverage_percent:.1f}%)")
    print(f"Incomplete/retryable records: {incomplete_count:,}")
    print(f"Summary CSV: {SUMMARY_FILE}")
    print(f"Review CSV: {REVIEW_FILE}")
    if not review.empty:
        print("\n========== FIRST 25 INCOMPLETE RECORDS ==========")
        print(review.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
