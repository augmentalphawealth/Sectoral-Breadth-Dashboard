from pathlib import Path
import time

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

    pd.DataFrame(
        [
            {
                "symbol": "HDFCBANK",
                "company_name": "HDFC BANK LTD",
                "isin": "INE040A01034",
            }
        ]
    ).to_csv(INPUT_FILE, index=False)


def clean(value):
    return "" if value is None else str(value).strip()


def main():
    ensure_input_file()
    nse_df = pd.read_csv(INPUT_FILE, dtype=str).fillna("")

    bse = BSE(download_folder=str(DOWNLOAD_FOLDER))

    try:
        rows = []

        for idx, row in nse_df.iterrows():
            symbol = clean(row.get("symbol"))
            company_name = clean(row.get("company_name"))
            isin = clean(row.get("isin"))

            print(f"[{idx + 1}/{len(nse_df)}] Looking up {symbol} ({isin})...")

            bse_code = ""
            meta = {}

            try:
                lookup = bse.lookup(isin)
                bse_code = clean(lookup.get("bse_code"))

                if not bse_code:
                    print("  No BSE code returned.")
                else:
                    print(f"  BSE code: {bse_code}")
                    meta = bse.equityMetaInfo(bse_code)
                    time.sleep(0.2)

            except Exception as exc:
                print(f"  Failed: {exc}")

            rows.append(
                {
                    "symbol": symbol,
                    "company_name": company_name,
                    "isin": isin,
                    "bse_code": bse_code,
                    "bse_sector": clean(meta.get("Sector")),
                    "bse_industry": clean(meta.get("Industry")),
                    "bse_industry_new": clean(meta.get("IndustryNew")),
                    "bse_i_group": clean(meta.get("IGroup")),
                    "bse_i_sub_group": clean(meta.get("ISubGroup")),
                }
            )

        output_df = pd.DataFrame(rows)
        output_df.to_csv(OUTPUT_FILE, index=False)

        matched = output_df["bse_sector"].ne("").sum()
        print(f"\nSaved industry mapping to: {OUTPUT_FILE}")
        print(f"Rows: {len(output_df)} | BSE sector matched: {matched}")

    finally:
        try:
            bse.exit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
