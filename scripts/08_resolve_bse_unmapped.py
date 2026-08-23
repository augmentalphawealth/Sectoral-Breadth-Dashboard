from pathlib import Path
import time

import pandas as pd
from bse import BSE


ROOT = Path(__file__).resolve().parents[1]

MASTER_FILE = ROOT / "data" / "processed" / "nse_mainboard_master_bse_classified.parquet"
MAPPING_FILE = ROOT / "data" / "processed" / "nse_bse_industry_mapping.csv"
UNMAPPED_FILE = ROOT / "data" / "processed" / "nse_bse_unmapped.csv"
STILL_UNMAPPED_FILE = ROOT / "data" / "processed" / "nse_bse_still_unmapped.csv"

DOWNLOAD_FOLDER = ROOT / "data" / "bse_downloads"
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 100
REQUEST_DELAY_SECONDS = 0.30

MAPPING_COLUMNS = [
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
    "classification_source",
    "bse_lookup_method",
]


def clean(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def load_mapping():
    if not MAPPING_FILE.exists():
        return pd.DataFrame(columns=MAPPING_COLUMNS)

    mapping_df = pd.read_csv(MAPPING_FILE, dtype=str).fillna("")

    for column in MAPPING_COLUMNS:
        if column not in mapping_df.columns:
            mapping_df[column] = ""

    mapping_df["row_number"] = pd.to_numeric(
        mapping_df["row_number"],
        errors="coerce",
    ).astype("Int64")

    return mapping_df[MAPPING_COLUMNS]


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

    master_df.to_parquet(MASTER_FILE, index=False)
    mapping_df.to_csv(MAPPING_FILE, index=False)


def find_bse_record(bse, isin, symbol, company_name):
    attempts = [
        ("ISIN", isin),
        ("SYMBOL", symbol),
        ("COMPANY_NAME", company_name),
    ]

    last_error = ""

    for method, query in attempts:
        if not query:
            continue

        try:
            result = bse.lookup(query) or {}
            bse_code = clean(result.get("bse_code"))

            if bse_code:
                return bse_code, method, ""

        except Exception as exc:
            last_error = f"{method}: {type(exc).__name__}: {exc}"

    if last_error:
        return "", "", last_error

    return "", "", "No BSE code found using ISIN, symbol, or company name"


def main():
    if not MASTER_FILE.exists():
        raise FileNotFoundError(f"Missing master file: {MASTER_FILE}")

    if not UNMAPPED_FILE.exists():
        raise FileNotFoundError(
            f"Missing unmapped report: {UNMAPPED_FILE}. Run workflow 07 first."
        )

    master_df = pd.read_parquet(MASTER_FILE)
    unmapped_df = pd.read_csv(UNMAPPED_FILE, dtype=str).fillna("")
    mapping_df = load_mapping()

    if "fallback_bse_attempted" not in master_df.columns:
        master_df["fallback_bse_attempted"] = ""

    if "classification_failure_reason" not in master_df.columns:
        master_df["classification_failure_reason"] = ""

    attempted_isins = set(
        master_df.loc[
            master_df["fallback_bse_attempted"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("YES"),
            "isin",
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    pending = unmapped_df[
        ~unmapped_df["isin"].fillna("").astype(str).str.strip().isin(
            attempted_isins
        )
    ].copy()

    pending = pending.head(BATCH_SIZE)

    print(f"Total initial BSE-unmapped stocks: {len(unmapped_df)}")
    print(f"Fallback BSE candidates in this batch: {len(pending)}")

    if pending.empty:
        still_unmapped = master_df[
            master_df["classification_status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("NOT_FOUND")
        ].copy()

        still_unmapped.to_csv(STILL_UNMAPPED_FILE, index=False)

        print("========== BSE FALLBACK COMPLETE ==========")
        print("No unattempted fallback candidates remain.")
        print(f"Still unmapped: {len(still_unmapped)}")
        print(f"Saved: {STILL_UNMAPPED_FILE}")
        return

    bse = BSE(download_folder=str(DOWNLOAD_FOLDER))
    new_mapping_rows = []

    try:
        for count, (_, candidate) in enumerate(pending.iterrows(), start=1):
            isin = clean(candidate.get("isin"))
            symbol = clean(candidate.get("symbol"))
            company_name = clean(candidate.get("company_name"))

            matches = master_df.index[
                master_df["isin"]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq(isin)
            ].tolist()

            if not matches:
                print(f"[{count}/{len(pending)}] {symbol} | ISIN missing in master")
                continue

            master_index = matches[0]
            row_number = master_df.index.get_loc(master_index) + 1

            print(f"[{count}/{len(pending)}] row {row_number} | {symbol} | {isin}")

            bse_code, lookup_method, failure_reason = find_bse_record(
                bse=bse,
                isin=isin,
                symbol=symbol,
                company_name=company_name,
            )

            meta = {}

            if bse_code:
                try:
                    meta = bse.equityMetaInfo(bse_code) or {}
                except Exception as exc:
                    failure_reason = (
                        f"equityMetaInfo {type(exc).__name__}: {exc}"
                    )

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

            classified = bool(sector)
            status = "CLASSIFIED" if classified else "NOT_FOUND"

            master_df.at[master_index, "fallback_bse_attempted"] = "YES"
            master_df.at[master_index, "classification_failure_reason"] = (
                "" if classified else failure_reason
            )

            if classified:
                master_df.at[master_index, "sector"] = sector
                master_df.at[master_index, "industry"] = industry
                master_df.at[master_index, "basic_industry"] = basic_industry
                master_df.at[master_index, "classification_status"] = "CLASSIFIED"
                master_df.at[
                    master_index,
                    "classification_source",
                ] = "BSE equityMetaInfo fallback"

                print(
                    f"  RESOLVED via {lookup_method} | BSE {bse_code} | "
                    f"Sector: {sector} | "
                    f"Industry: {clean(meta.get('Industry'))}"
                )
            else:
                print(f"  Still unresolved: {failure_reason}")

            new_mapping_rows.append(
                {
                    "row_number": row_number,
                    "symbol": symbol,
                    "company_name": company_name,
                    "series": clean(master_df.at[master_index, "series"]),
                    "isin": isin,
                    "bse_code": bse_code,
                    "bse_sector": sector,
                    "bse_industry": clean(meta.get("Industry")),
                    "bse_industry_new": clean(meta.get("IndustryNew")),
                    "bse_i_group": clean(meta.get("IGroup")),
                    "bse_i_sub_group": clean(meta.get("ISubGroup")),
                    "classification_status": status,
                    "classification_source": (
                        "BSE equityMetaInfo fallback" if classified else ""
                    ),
                    "bse_lookup_method": lookup_method,
                }
            )

            time.sleep(REQUEST_DELAY_SECONDS)

    finally:
        try:
            bse.exit()
        except Exception:
            pass

    if new_mapping_rows:
        mapping_df = pd.concat(
            [mapping_df, pd.DataFrame(new_mapping_rows)],
            ignore_index=True,
        )

    save_outputs(master_df, mapping_df)

    still_unmapped = master_df[
        master_df["classification_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("NOT_FOUND")
    ].copy()

    still_unmapped.to_csv(STILL_UNMAPPED_FILE, index=False)

    resolved = sum(
        row["classification_status"] == "CLASSIFIED"
        for row in new_mapping_rows
    )

    remaining_fallback = master_df[
        master_df["classification_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("NOT_FOUND")
        & ~master_df["fallback_bse_attempted"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("YES")
    ]

    print("\n========== BSE FALLBACK BATCH COMPLETE ==========")
    print(f"Fallback candidates processed this run: {len(new_mapping_rows)}")
    print(f"Resolved through BSE fallback: {resolved}")
    print(f"Still unresolved total: {len(still_unmapped)}")
    print(f"Fallback candidates not yet attempted: {len(remaining_fallback)}")
    print(f"Updated master: {MASTER_FILE}")
    print(f"Updated mapping: {MAPPING_FILE}")
    print(f"Still-unmapped report: {STILL_UNMAPPED_FILE}")


if __name__ == "__main__":
    main()
