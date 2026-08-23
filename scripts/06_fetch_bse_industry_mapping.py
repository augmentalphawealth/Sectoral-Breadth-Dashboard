from pathlib import Path
import os
import time

import pandas as pd
from bse import BSE


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "nse_mainboard_master.parquet"
OUTPUT_FILE = ROOT / "data" / "processed" / "nse_mainboard_master_bse_classified.parquet"
MAPPING_FILE = ROOT / "data" / "processed" / "nse_bse_industry_mapping.csv"

DOWNLOAD_FOLDER = ROOT / "data" / "bse_downloads"
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

START_INDEX = int(os.getenv("START_INDEX", "0"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
REQUEST_DELAY_SECONDS = 0.30


def clean(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def load_output():
    if OUTPUT_FILE.exists():
        df = pd.read_parquet(OUTPUT_FILE)
        print(f"Resuming from existing output: {OUTPUT_FILE}")
        return df

    df = pd.read_parquet(INPUT_FILE)

    for column in [
        "industry",
        "basic_industry",
        "sector",
        "classification_status",
        "classification_source",
    ]:
        if column not in df.columns:
            df[column] = ""

    return df


def main():
    df = load_output()

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

    total = len(df)
    end_index = min(START_INDEX + BATCH_SIZE, total)

    if START_INDEX >= total:
        print(f"Nothing to process. START_INDEX={START_INDEX}, total rows={total}.")
        return

    print(f"Processing rows {START_INDEX + 1} through {end_index} of {total}.")

    bse = BSE(download_folder=str(DOWNLOAD_FOLDER))
    batch_rows = []

    try:
        for position in range(START_INDEX, end_index):
            row = df.iloc[position]

            symbol = clean(row["symbol"])
            company_name = clean(row["company_name"])
            isin = clean(row["isin"])

            print(f"[{position + 1}/{total}] {symbol} | {isin}")

            bse_code = ""
            meta = {}

            try:
                lookup = bse.lookup(isin) or {}
                bse_code = clean(lookup.get("bse_code"))

                if bse_code:
                    meta = bse.equityMetaInfo(bse_code) or {}

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

            if not industry:
                industry = clean(meta.get("IndustryNew")) or clean(meta.get("Industry"))

            if not basic_industry:
                basic_industry = clean(meta.get("Industry"))

            df.at[df.index[position], "sector"] = sector
            df.at[df.index[position], "industry"] = industry
            df.at[df.index[position], "basic_industry"] = basic_industry
            df.at[df.index[position], "classification_status"] = (
                "CLASSIFIED" if sector else "NOT_FOUND"
            )
            df.at[df.index[position], "classification_source"] = (
                "BSE equityMetaInfo" if sector else ""
            )

            batch_rows.append(
                {
                    "row_number": position + 1,
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
                    "classification_status": (
                        "CLASSIFIED" if sector else "NOT_FOUND"
                    ),
                }
            )

            time.sleep(REQUEST_DELAY_SECONDS)

    finally:
        try:
            bse.exit()
        except Exception:
            pass

    df.to_parquet(OUTPUT_FILE, index=False)

    batch_df = pd.DataFrame(batch_rows)
    if MAPPING_FILE.exists():
        existing_mapping = pd.read_csv(MAPPING_FILE, dtype=str).fillna("")
        mapping_df = pd.concat([existing_mapping, batch_df], ignore_index=True)
        mapping_df = mapping_df.drop_duplicates(
            subset=["row_number"], keep="last"
        ).sort_values("row_number")
    else:
        mapping_df = batch_df

    mapping_df.to_csv(MAPPING_FILE, index=False)

    classified = (df["classification_status"] == "CLASSIFIED").sum()
    not_found = (df["classification_status"] == "NOT_FOUND").sum()

    print("\n========== BATCH COMPLETE ==========")
    print(f"Processed rows: {START_INDEX + 1} to {end_index}")
    print(f"Next START_INDEX: {end_index}")
    print(f"Total classified so far: {classified}")
    print(f"Total not found so far: {not_found}")
    print(f"Updated master: {OUTPUT_FILE}")
    print(f"Updated mapping CSV: {MAPPING_FILE}")


if __name__ == "__main__":
    main()
