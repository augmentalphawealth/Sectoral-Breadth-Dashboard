from pathlib import Path

import pandas as pd
from bse import BSE


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_FOLDER = ROOT / "data" / "bse_downloads"
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

INPUT_FILE = ROOT / "data" / "nse_mainboard_companies.csv"
OUTPUT_FILE = ROOT / "data" / "nse_industry_mapping.csv"


def ensure_input_file():
    if INPUT_FILE.exists():
        return

    print(f"Input file not found: {INPUT_FILE}")
    print("Creating a minimal starter file with HDFCBANK...")

    INPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    starter = pd.DataFrame(
        [
            {
                "symbol": "HDFCBANK",
                "company_name": "HDFC BANK LTD",
                "isin": "INE040A01034",
            }
        ]
    )
    starter.to_csv(INPUT_FILE, index=False)
    print(f"Created: {INPUT_FILE}")


def main():
    ensure_input_file()

    bse = BSE(download_folder=str(DOWNLOAD_FOLDER))

    try:
        nse_df = pd.read_csv(INPUT_FILE)

        rows = []
        for idx, row in nse_df.iterrows():
            symbol = row["symbol"]
            company_name = row["company_name"]
            isin = row["isin"]

            print(f"[{idx + 1}/{len(nse_df)}] Fetching BSE metadata for {symbol} ({isin})...")

            try:
                meta = bse.equityMetaInfo(isin)
            except Exception as e:
                print(f"  equityMetaInfo({isin}) failed: {e}")
                meta = {}

            sector = meta.get("Sector", "")
            industry = meta.get("Industry", "")
            industry_new = meta.get("IndustryNew", "")
            i_group = meta.get("IGroup", "")
            i_sub_group = meta.get("ISubGroup", "")

            rows.append(
                {
                    "symbol": symbol,
                    "company_name": company_name,
                    "isin": isin,
                    "bse_sector": sector,
                    "bse_industry": industry,
                    "bse_industry_new": industry_new,
                    "bse_i_group": i_group,
                    "bse_i_sub_group": i_sub_group,
                }
            )

        out_df = pd.DataFrame(rows)
        out_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved industry mapping to: {OUTPUT_FILE}")

    finally:
        try:
            bse.exit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
