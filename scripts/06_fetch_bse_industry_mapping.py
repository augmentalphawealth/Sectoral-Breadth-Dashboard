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

CHECKPOINT_SIZE = 100
REQUEST_DELAY_SECONDS = 0.30


def clean(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def load_master():
    if OUTPUT_FILE.exists():
        print(f"Resuming from saved output: {OUTPUT_FILE}")
        master_df = pd.read_parquet(OUTPUT_FILE)
    else:
        print(f"Starting from master source: {INPUT_FILE}")
        master_df = pd.read_parquet(INPUT_FILE)

    required_classification_columns = [
        "industry",
        "basic_industry",
        "sector",
        "classification_status",
        "classification_source",
    ]

    for column in required_classification_columns:
        if column not in master_df.columns:
            master_df[column] = ""

    return master_df


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

    mapping_df = pd.read_csv(MAPPING_FILE, dtype=str).fillna("")

    for column in columns:
        if column not in mapping_df.columns:
            mapping_df[column] = ""

    mapping_df["row_number"] = pd.to_numeric(
        mapping_df["row_number"],
        errors="coerce",
    ).astype("Int64")

    return mapping_df[columns]


def save_checkpoint(master_df, mapping_df):
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

    return mapping_df


def is_processed(row):
    return clean(row.get("classification_status")) in {
        "CLASSIFIED",
        "NOT_FOUND",
    }


def main():
    master_df = load_master()
    mapping_df = load_mapping()

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

    missing = [
        column for column in required_columns if column not in master_df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    pending_positions = [
        position
        for position, (_, row) in enumerate(master_df.iterrows())
        if not is_processed(row)
    ]

    total = len(master_df)
    pending_total = len(pending_positions)

    print(f"Total securities: {total}")
    print(f"Already processed: {total - pending_total}")
    print(f"Pending: {pending_total}")

    if not pending_positions:
        print("All securities have already been processed.")
        return

    bse = BSE(download_folder=str(DOWNLOAD_FOLDER))
    checkpoint_rows = []

    try:
        for processed_count, position in enumerate(pending_positions, start=1):
            row = master_df.iloc[position]

            symbol = clean(row["symbol"])
            company_name = clean(row["company_name"])
            isin = clean(row["isin"])

            print(
                f"[{processed_count}/{pending_total}] "
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

            checkpoint_rows.append(
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

            if processed_count % CHECKPOINT_SIZE == 0:
                checkpoint_df = pd.DataFrame(checkpoint_rows)

                mapping_df = pd.concat(
                    [mapping_df, checkpoint_df],
                    ignore_index=True,
                )

                mapping_df = save_checkpoint(master_df, mapping_df)
                checkpoint_rows = []

                classified = (
                    master_df["classification_status"] == "CLASSIFIED"
                ).sum()

                not_found = (
                    master_df["classification_status"] == "NOT_FOUND"
                ).sum()

                print("\n========== CHECKPOINT SAVED ==========")
                print(f"Completed in this run: {processed_count}/{pending_total}")
                print(f"Total classified: {classified}")
                print(f"Total not found: {not_found}")
                print(f"Saved master: {OUTPUT_FILE}")
                print(f"Saved mapping: {MAPPING_FILE}\n")

            time.sleep(REQUEST_DELAY_SECONDS)

    finally:
        try:
            bse.exit()
        except Exception:
            pass

    if checkpoint_rows:
        checkpoint_df = pd.DataFrame(checkpoint_rows)

        mapping_df = pd.concat(
            [mapping_df, checkpoint_df],
            ignore_index=True,
        )

        mapping_df = save_checkpoint(master_df, mapping_df)

        print("\n========== FINAL PARTIAL CHECKPOINT SAVED ==========")
        print(f"Saved remaining records: {len(checkpoint_rows)}")

    classified = (master_df["classification_status"] == "CLASSIFIED").sum()
    not_found = (master_df["classification_status"] == "NOT_FOUND").sum()

    print("\n========== COMPLETE ==========")
    print(f"Total securities: {len(master_df)}")
    print(f"Total classified: {classified}")
    print(f"Total not found: {not_found}")
    print(f"Saved master: {OUTPUT_FILE}")
    print(f"Saved mapping: {MAPPING_FILE}")


if __name__ == "__main__":
    main()
