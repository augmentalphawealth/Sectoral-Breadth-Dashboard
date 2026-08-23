from pathlib import Path
import time

import pandas as pd
from bse import BSE


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "nse_mainboard_master.parquet"
OUTPUT_FILE = ROOT / "data" / "processed" / "nse_mainboard_master_bse_classified.parquet"
CSV_OUTPUT_FILE = ROOT / "data" / "processed" / "nse_bse_industry_mapping.csv"

DOWNLOAD_FOLDER = ROOT / "data" / "bse_downloads"
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

REQUEST_DELAY_SECONDS = 0.25


def clean(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def main():
    df = pd.read_parquet(INPUT_FILE)

    required_columns = [
        "symbol",
        "company_name",
        "isin",
        "industry",
        "basic_industry",
        "sector",
        "classification_status",
        "classification_source",
    ]

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    bse = BSE(download_folder=str(DOWNLOAD_FOLDER))
    rows = []

    try:
        total = len(df)

        for position, (index, row) in enumerate(df.iterrows(), start=1):
            symbol = clean(row["symbol"])
            company_name = clean(row["company_name"])
            isin = clean(row["isin"])

            print(f"[{position}/{total}] {symbol} | {isin}")

            bse_code = ""
            meta = {}

            try:
                lookup = bse.lookup(isin)
                bse_code = clean(lookup.get("bse_code"))

                if bse_code:
                    meta = bse.equityMetaInfo(bse_code)
                    print(
                        f"  BSE {bse_code} | "
                        f"Sector: {clean(meta.get('Sector'))} | "
                        f"Industry: {clean(meta.get('Industry'))}"
                    )
                else:
                    print("  No BSE code found.")

            except Exception as exc:
                print(f"  Failed: {type(exc).__name__}: {exc}")

            sector = clean(meta.get("Sector"))
            industry = clean(meta.get("IGroup"))
            basic_industry = clean(meta.get("ISubGroup"))

            # Fall back to BSE's older Industry field when group fields are absent.
            if not industry:
                industry = clean(meta.get("IndustryNew")) or clean(meta.get("Industry"))
            if not basic_industry:
                basic_industry = clean(meta.get("Industry"))

            df.at[index, "sector"] = sector
            df.at[index, "industry"] = industry
            df.at[index, "basic_industry"] = basic_industry
            df.at[index, "classification_status"] = (
                "CLASSIFIED" if sector else "NOT_FOUND"
            )
            df.at[index, "classification_source"] = (
                "BSE equityMetaInfo" if sector else ""
            )

            rows.append(
                {
                    "symbol": symbol,
                    "company_name": company_name,
                    "series": clean(row.get("series")),
                    "isin": isin,
                    "bse_code": bse_code,
                    "bse_sector": sector,
                    "bse_industry": clean(meta.get("Industry")),
                    "bse_industry_new": clean(meta.get("IndustryNew")),
                    "bse_i_group": clean(meta.get("IGroup")),
                    "bse_i_sub_group": clean(meta.get("ISubGroup")),
                    "classification_status": df.at[index, "classification_status"],
                }
            )

            time.sleep(REQUEST_DELAY_SECONDS)

    finally:
        try:
            bse.exit()
        except Exception:
            pass

    mapping_df = pd.DataFrame(rows)

    df.to_parquet(OUTPUT_FILE, index=False)
    mapping_df.to_csv(CSV_OUTPUT_FILE, index=False)

    classified = (df["classification_status"] == "CLASSIFIED").sum()
    not_found = (df["classification_status"] == "NOT_FOUND").sum()

    print("\n========== COMPLETE ==========")
    print(f"Input rows: {len(df)}")
    print(f"Classified: {classified}")
    print(f"Not found: {not_found}")
    print(f"Updated master: {OUTPUT_FILE}")
    print(f"BSE mapping CSV: {CSV_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
