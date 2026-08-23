from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MASTER_FILE = ROOT / "data" / "processed" / "nse_mainboard_master_bse_classified.parquet"
SUMMARY_FILE = ROOT / "data" / "processed" / "nse_classification_summary.csv"
REVIEW_FILE = ROOT / "data" / "processed" / "nse_classification_review.csv"


def clean(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def main():
    df = pd.read_parquet(MASTER_FILE)

    for column in [
        "classification_status",
        "classification_source",
        "sector",
        "industry",
        "basic_industry",
        "yahoo_sector",
        "yahoo_industry",
    ]:
        if column not in df.columns:
            df[column] = ""

    df["classification_status"] = (
        df["classification_status"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    summary = (
        df.groupby(
            ["classification_status", "classification_source"],
            dropna=False,
        )
        .size()
        .reset_index(name="stock_count")
        .sort_values(
            ["classification_status", "classification_source"],
        )
    )

    summary.to_csv(SUMMARY_FILE, index=False)

    review_columns = [
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
        "yahoo_ticker",
        "yahoo_sector",
        "yahoo_industry",
        "classification_failure_reason",
        "yahoo_failure_reason",
    ]

    review_columns = [
        column for column in review_columns if column in df.columns
    ]

    review = df[
        df["classification_status"].isin(
            ["YAHOO_FALLBACK", "NOT_FOUND"]
        )
    ][review_columns].copy()

    review = review.sort_values(
        ["classification_status", "symbol"],
    )

    review.to_csv(REVIEW_FILE, index=False)

    total = len(df)
    bse_count = int((df["classification_status"] == "CLASSIFIED").sum())
    yahoo_count = int(
        (df["classification_status"] == "YAHOO_FALLBACK").sum()
    )
    not_found_count = int(
        (df["classification_status"] == "NOT_FOUND").sum()
    )

    print("========== CLASSIFICATION AUDIT COMPLETE ==========")
    print(f"Total stocks: {total}")
    print(f"BSE classified: {bse_count}")
    print(f"Yahoo fallback: {yahoo_count}")
    print(f"Not found: {not_found_count}")
    print(f"Summary CSV: {SUMMARY_FILE}")
    print(f"Review CSV: {REVIEW_FILE}")


if __name__ == "__main__":
    main()
