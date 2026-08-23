from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "nse_mainboard_master_bse_classified.parquet"
OUTPUT_FILE = ROOT / "data" / "processed" / "nse_bse_unmapped.csv"


def main():
    df = pd.read_parquet(INPUT_FILE)

    unmapped = df[
        df["classification_status"].fillna("").eq("NOT_FOUND")
    ].copy()

    columns = [
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
    ]

    columns = [column for column in columns if column in unmapped.columns]
    unmapped = unmapped[columns].sort_values(["symbol", "isin"])

    unmapped.to_csv(OUTPUT_FILE, index=False)

    print("========== UNMAPPED REPORT COMPLETE ==========")
    print(f"Total stocks: {len(df)}")
    print(f"Unmapped stocks: {len(unmapped)}")
    print(f"Saved report: {OUTPUT_FILE}")

    if not unmapped.empty:
        print("\n========== FIRST 25 UNMAPPED ==========")
        print(unmapped.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
