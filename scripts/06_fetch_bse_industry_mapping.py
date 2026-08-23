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
        df = pd.read_parquet(OUTPUT_FILE)
    else:
        print(f"Starting from master source: {INPUT_FILE}")
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
    status = clean(row.get("classification_status"))
    return status in {"CLASSIFIED", "NOT_FOUND"}


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

    missing = [column for column in required_columns if column not in master_df.columns]
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
