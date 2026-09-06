# scripts/06_fetch_bse_industry_mapping.py
# Primary BSE classification pass for new and incomplete NSE securities.
# It preserves complete existing mappings and records failures as retryable.

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import pandas as pd
from bse import BSE

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
INPUT_FILE = PROCESSED / "nse_mainboard_master.parquet"
OUTPUT_FILE = PROCESSED / "nse_mainboard_master_bse_classified.parquet"
MAPPING_FILE = PROCESSED / "nse_bse_industry_mapping.csv"
DOWNLOAD_FOLDER = ROOT / "data" / "bse_downloads"
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 100
REQUEST_DELAY_SECONDS = 0.30
RETRYABLE_STATUSES = {"PENDING", "REVIEW_REQUIRED", "NOT_FOUND", "BSE_RETRY"}

MASTER_COLUMNS = [
    "symbol", "company_name", "series", "isin", "listing_date",
    "sector", "industry", "basic_industry", "classification_status",
    "classification_source", "classification_failure_reason",
    "bse_attempt_count", "bse_last_attempt_utc", "bse_code",
]

MAPPING_COLUMNS = [
    "isin", "symbol", "company_name", "series", "bse_code",
    "bse_sector", "bse_industry", "bse_industry_new", "bse_i_group",
    "bse_i_sub_group", "classification_status", "classification_source",
    "failure_reason", "attempted_at_utc", "attempt_count",
]


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def ensure_master_schema(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in MASTER_COLUMNS:
        if column not in data.columns:
            data[column] = ""
    for column in MASTER_COLUMNS:
        if column != "listing_date":
            data[column] = text_series(data[column])
    data["listing_date"] = pd.to_datetime(data["listing_date"], errors="coerce")
    data["bse_attempt_count"] = pd.to_numeric(data["bse_attempt_count"], errors="coerce").fillna(0).astype(int)
    return data


def load_master() -> pd.DataFrame:
    source = OUTPUT_FILE if OUTPUT_FILE.exists() else INPUT_FILE
    if not source.exists():
        raise FileNotFoundError(f"Missing NSE master input: {source}")
    print(f"Loading master: {source}")
    return ensure_master_schema(pd.read_parquet(source))


def load_mapping() -> pd.DataFrame:
    if not MAPPING_FILE.exists():
        return pd.DataFrame(columns=MAPPING_COLUMNS)
    data = pd.read_csv(MAPPING_FILE, dtype=str).fillna("")
    for column in MAPPING_COLUMNS:
        if column not in data.columns:
            data[column] = ""
    data["attempt_count"] = pd.to_numeric(data["attempt_count"], errors="coerce").fillna(0).astype(int)
    return data[MAPPING_COLUMNS]


def complete_mapping_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["sector"].ne("") & frame["industry"].ne("") & frame["basic_industry"].ne("")


def eligible_mask(frame: pd.DataFrame) -> pd.Series:
    status = text_series(frame["classification_status"])
    has_identity = frame["symbol"].ne("") & frame["isin"].ne("")
    return has_identity & ~complete_mapping_mask(frame) & status.isin(RETRYABLE_STATUSES)


def extract_classification(meta: dict) -> tuple[str, str, str]:
    sector = clean(meta.get("Sector"))
    industry = clean(meta.get("IGroup")) or clean(meta.get("IndustryNew")) or clean(meta.get("Industry"))
    basic_industry = clean(meta.get("ISubGroup")) or clean(meta.get("Industry"))
    return sector, industry, basic_industry


def attempt_bse_classification(bse: BSE, isin: str) -> tuple[str, dict, str]:
    try:
        lookup = bse.lookup(isin) or {}
        bse_code = clean(lookup.get("bse_code"))
        if not bse_code:
            return "", {}, "No BSE code found for ISIN"
        metadata = bse.equityMetaInfo(bse_code) or {}
        if not isinstance(metadata, dict):
            return bse_code, {}, "BSE metadata response was not a dictionary"
        return bse_code, metadata, ""
    except Exception as exc:
        return "", {}, f"{type(exc).__name__}: {exc}"


def save_outputs(master: pd.DataFrame, mapping: pd.DataFrame) -> None:
    master = ensure_master_schema(master)
    master = master.drop_duplicates(subset=["isin"], keep="last").sort_values(["symbol", "series", "isin"]).reset_index(drop=True)
    mapping = mapping.copy()
    if not mapping.empty:
        mapping["attempt_count"] = pd.to_numeric(mapping["attempt_count"], errors="coerce").fillna(0).astype(int)
        mapping = mapping.drop_duplicates(subset=["isin"], keep="last").sort_values(["symbol", "isin"]).reset_index(drop=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(OUTPUT_FILE, index=False)
    mapping.to_csv(MAPPING_FILE, index=False)


def main() -> None:
    print("========== BSE PRIMARY CLASSIFICATION START ==========")
    master = load_master()
    mapping = load_mapping()

    pending = master.loc[eligible_mask(master)].copy()
    pending = pending.sort_values(["bse_attempt_count", "symbol", "isin"]).head(BATCH_SIZE)

    print(f"Master rows: {len(master):,}")
    print(f"Complete mappings retained: {int(complete_mapping_mask(master).sum()):,}")
    print(f"Eligible BSE retries before batch: {int(eligible_mask(master).sum()):,}")
    print(f"Processing this run: {len(pending):,}")

    if pending.empty:
        save_outputs(master, mapping)
        print("No pending or incomplete records need a BSE primary lookup.")
        return

    now_text = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    new_mapping_rows: list[dict] = []
    bse = BSE(download_folder=str(DOWNLOAD_FOLDER))

    try:
        for number, (index, row) in enumerate(pending.iterrows(), start=1):
            symbol = clean(row["symbol"])
            isin = clean(row["isin"])
            previous_attempts = int(master.at[index, "bse_attempt_count"] or 0)
            print(f"[{number}/{len(pending)}] {symbol} | {isin} | prior attempts: {previous_attempts}")

            bse_code, meta, failure_reason = attempt_bse_classification(bse, isin)
            sector, industry, basic_industry = extract_classification(meta)
            complete = bool(sector and industry and basic_industry)

            master.at[index, "bse_attempt_count"] = previous_attempts + 1
            master.at[index, "bse_last_attempt_utc"] = now_text
            master.at[index, "bse_code"] = bse_code

            if complete:
                master.at[index, "sector"] = sector
                master.at[index, "industry"] = industry
                master.at[index, "basic_industry"] = basic_industry
                master.at[index, "classification_status"] = "CLASSIFIED"
                master.at[index, "classification_source"] = "BSE equityMetaInfo"
                master.at[index, "classification_failure_reason"] = ""
                status = "CLASSIFIED"
                source = "BSE equityMetaInfo"
                reason = ""
                print(f"  Classified | {sector} | {industry} | {basic_industry}")
            else:
                # Do not erase any partial valid source data and do not treat a
                # temporary BSE failure as a permanent classification decision.
                master.at[index, "classification_status"] = "BSE_RETRY"
                master.at[index, "classification_failure_reason"] = failure_reason or "BSE returned incomplete hierarchy"
                status = "BSE_RETRY"
                source = ""
                reason = master.at[index, "classification_failure_reason"]
                print(f"  Retry later | {reason}")

            new_mapping_rows.append({
                "isin": isin,
                "symbol": symbol,
                "company_name": clean(row["company_name"]),
                "series": clean(row["series"]),
                "bse_code": bse_code,
                "bse_sector": sector,
                "bse_industry": clean(meta.get("Industry")),
                "bse_industry_new": clean(meta.get("IndustryNew")),
                "bse_i_group": clean(meta.get("IGroup")),
                "bse_i_sub_group": clean(meta.get("ISubGroup")),
                "classification_status": status,
                "classification_source": source,
                "failure_reason": reason,
                "attempted_at_utc": now_text,
                "attempt_count": previous_attempts + 1,
            })
            time.sleep(REQUEST_DELAY_SECONDS)
    finally:
        try:
            bse.exit()
        except Exception:
            pass

    mapping = pd.concat([mapping, pd.DataFrame(new_mapping_rows)], ignore_index=True)
    save_outputs(master, mapping)

    completed = int(complete_mapping_mask(master).sum())
    retrying = int(text_series(master["classification_status"]).eq("BSE_RETRY").sum())
    print("========== BSE PRIMARY CLASSIFICATION COMPLETE ==========")
    print(f"Complete classifications: {completed:,}")
    print(f"Retryable BSE records: {retrying:,}")
    print(f"Master output: {OUTPUT_FILE}")
    print(f"Mapping audit: {MAPPING_FILE}")


if __name__ == "__main__":
    main()
