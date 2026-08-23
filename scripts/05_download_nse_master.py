from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NSE_EQUITY_MASTER_URL = (
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
)


def main():
    print("Downloading NSE equity master...")

    df = pd.read_csv(NSE_EQUITY_MASTER_URL)
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

    # Current company-share candidates. BE/BZ stay included because
    # they are part of your NSE company universe and may use Angel fallback upstream.
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
