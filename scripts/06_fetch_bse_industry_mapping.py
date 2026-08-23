from pathlib import Path
import time

import pandas as pd
from bse import BSE


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "nse_mainboard_master.parquet"
OUTPUT_FILE = ROOT / "data" / "processed" / "nse_mainboard_master_bse_classified.parquet"
MAPPING_FILE = ROOT / "data" / "processed" / "nse_bse_industry_mapping.csv"

DOWNLOAD_FOLDER = ROOT / "data" / "bse_downloads"
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 100
REQUEST_DELAY_SECONDS = 0.30


def clean(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def load_master():
    if OUTPUT_FILE.exists():
        print(f"Resuming from: {OUTPUT_FILE}")
        df = pd.read_parquet(OUTPUT_FILE)
    else:
        print(f"Starting from: {INPUT_FILE}")
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


def load_mapping():
    columns = [
        "row_number",
        "symbol",
        "company_name",
        "series",
        "isin",
        "bse_code",
        "bse_sector",
        "bse_industry",
        "bse_industry_new",
        "bse_i_group",
        "bse_i_sub_group",
        "classification_status",
    ]

    if not MAPPING_FILE.exists():
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(MAPPING_FILE, dtype=str).fillna("")

    for column in columns:
        if column not in df.columns:
            df[column] = ""

    df["row_number"] = pd.to_numeric(
        df["row_number"],
        errors="coerce",
    ).astype("Int64")

    return df[columns]


def is_processed(row):
    return clean(row.get("classification_status")) in {
        "CLASSIFIED",
        "NOT_FOUND",
    }


def save_outputs(master_df, mapping_df):
    mapping_df["row_number"] = pd.to_numeric(
        mapping_df["row_number"],
        errors="coerce",
    ).astype("Int64")

    mapping_df = (
        mapping_df.drop_duplicates(subset=["row_number"], keep="last")
        .sort_values("row_number")
        .reset_index(drop=True)
    )

    master_df.to_parquet(OUTPUT_FILE, index=False)
    mapping_df.to_csv(MAPPING_FILE, index=False)


def main():
    master_df = load_master()
    mapping_df = load_mapping()

    pending_positions = [
        position
        for position, (_, row) in enumerate(master_df.iterrows())
        if not is_processed(row)
    ]

    total = len(master_df)
    already_processed = total - len(pending_positions)
    batch_positions = pending_positions[:BATCH_SIZE]

    print(f"Total securities: {total}")
    print(f"Already processed: {already_processed}")
    print(f"Pending before this batch: {len(pending_positions)}")

    if not batch_positions:
        print("========== ALL COMPLETE ==========")
        print("No pending securities remain.")
        return

    print(
        f"Processing this batch: {len(batch_positions)} securities "
        f"(rows {batch_positions[0] + 1} to {batch_positions[-1] + 1})."
    )

    bse = BSE(download_folder=str(DOWNLOAD_FOLDER))
    new_mapping_rows = []

    try:
        for batch_number, position in enumerate(batch_positions, start=1):
            row = master_df.iloc[position]

            symbol = clean(row["symbol"])
            company_name = clean(row["company_name"])
            isin = clean(row["isin"])

            print(
                f"[{batch_number}/{len(batch_positions)}] "
                f"row {position + 1}/{total} | {symbol} | {isin}"
            )

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
                industry = (
                    clean(meta.get("IndustryNew"))
                    or clean(meta.get("Industry"))
                )

            if not basic_industry:
                basic_industry = clean(meta.get("Industry"))

            status = "CLASSIFIED" if sector else "NOT_FOUND"
            master_index = master_df.index[position]

            master_df.at[master_index, "sector"] = sector
            master_df.at[master_index, "industry"] = industry
            master_df.at[master_index, "basic_industry"] = basic_industry
            master_df.at[master_index, "classification_status"] = status
            master_df.at[master_index, "classification_source"] = (
                "BSE equityMetaInfo" if sector else ""
            )

            new_mapping_rows.append(
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
                    "classification_status": status,
                }
            )

            time.sleep(REQUEST_DELAY_SECONDS)

    finally:
        try:
            bse.exit()
        except Exception:
            pass

    mapping_df = pd.concat(
        [mapping_df, pd.DataFrame(new_mapping_rows)],
        ignore_index=True,
    )

    save_outputs(master_df, mapping_df)

    classified = (master_df["classification_status"] == "CLASSIFIED").sum()
    not_found = (master_df["classification_status"] == "NOT_FOUND").sum()
    remaining = total - classified - not_found

    print("\n========== BATCH COMPLETE ==========")
    print(f"Processed this batch: {len(batch_positions)}")
    print(f"Total classified: {classified}")
    print(f"Total not found: {not_found}")
    print(f"Remaining: {remaining}")
    print(f"Saved master: {OUTPUT_FILE}")
    print(f"Saved mapping CSV: {MAPPING_FILE}")


if __name__ == "__main__":
    main()
