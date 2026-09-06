# scripts/08_resolve_bse_unmapped.py
# Secondary BSE lookup for incomplete classifications.
# Uses ISIN first, then symbol, then company name. It never guesses a mapping,
# never overwrites a complete hierarchy, and leaves failures retryable for the
# daily automation and Yahoo fallback.

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import pandas as pd
from bse import BSE

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MASTER_FILE = PROCESSED / "nse_mainboard_master_bse_classified.parquet"
MAPPING_FILE = PROCESSED / "nse_bse_industry_mapping.csv"
UNMAPPED_FILE = PROCESSED / "nse_bse_unmapped.csv"
STILL_UNMAPPED_FILE = PROCESSED / "nse_bse_still_unmapped.csv"
DOWNLOAD_FOLDER = ROOT / "data" / "bse_downloads"
DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 100
REQUEST_DELAY_SECONDS = 0.30
RETRYABLE_STATUSES = {"PENDING", "REVIEW_REQUIRED", "NOT_FOUND", "BSE_RETRY"}

MAPPING_COLUMNS = [
    "isin", "symbol", "company_name", "series", "bse_code",
    "bse_sector", "bse_industry", "bse_industry_new", "bse_i_group",
    "bse_i_sub_group", "classification_status", "classification_source",
    "failure_reason", "attempted_at_utc", "attempt_count", "bse_lookup_method",
]


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def ensure_master_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in [
        "symbol", "company_name", "series", "isin", "sector", "industry",
        "basic_industry", "classification_status", "classification_source",
        "classification_failure_reason", "fallback_bse_attempt_count",
        "fallback_bse_last_attempt_utc", "fallback_bse_attempted",
    ]:
        if column not in data.columns:
            data[column] = ""
        data[column] = text_series(data[column])
    data["fallback_bse_attempt_count"] = pd.to_numeric(
        data["fallback_bse_attempt_count"], errors="coerce"
    ).fillna(0).astype(int)
    return data


def complete_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["sector"].ne("") & frame["industry"].ne("") & frame["basic_industry"].ne("")


def load_mapping() -> pd.DataFrame:
    if not MAPPING_FILE.exists():
        return pd.DataFrame(columns=MAPPING_COLUMNS)
    mapping = pd.read_csv(MAPPING_FILE, dtype=str).fillna("")
    for column in MAPPING_COLUMNS:
        if column not in mapping.columns:
            mapping[column] = ""
    mapping["attempt_count"] = pd.to_numeric(mapping["attempt_count"], errors="coerce").fillna(0).astype(int)
    return mapping[MAPPING_COLUMNS]


def find_bse_record(bse: BSE, isin: str, symbol: str, company_name: str) -> tuple[str, str, str]:
    attempts = [("ISIN", isin), ("SYMBOL", symbol), ("COMPANY_NAME", company_name)]
    failures: list[str] = []
    for method, query in attempts:
        if not query:
            continue
        try:
            result = bse.lookup(query) or {}
            bse_code = clean(result.get("bse_code"))
            if bse_code:
                return bse_code, method, ""
        except Exception as exc:
            failures.append(f"{method}: {type(exc).__name__}: {exc}")
    if failures:
        return "", "", " | ".join(failures)
    return "", "", "No BSE code found using ISIN, symbol, or company name"


def extract_hierarchy(metadata: dict) -> tuple[str, str, str]:
    sector = clean(metadata.get("Sector"))
    industry = clean(metadata.get("IGroup")) or clean(metadata.get("IndustryNew")) or clean(metadata.get("Industry"))
    basic_industry = clean(metadata.get("ISubGroup")) or clean(metadata.get("Industry"))
    return sector, industry, basic_industry


def write_still_unmapped(master: pd.DataFrame) -> pd.DataFrame:
    incomplete = master[~complete_mask(master)].copy()
    incomplete = incomplete[text_series(incomplete["classification_status"]).isin(RETRYABLE_STATUSES | {"BSE_RETRY"})].copy()
    columns = [
        "symbol", "company_name", "series", "isin", "listing_date", "sector",
        "industry", "basic_industry", "classification_status", "classification_source",
        "classification_failure_reason", "fallback_bse_attempt_count",
        "fallback_bse_last_attempt_utc",
    ]
    columns = [column for column in columns if column in incomplete.columns]
    incomplete = incomplete[columns].drop_duplicates("isin", keep="last").sort_values(["symbol", "isin"])
    incomplete.to_csv(STILL_UNMAPPED_FILE, index=False)
    return incomplete


def main() -> None:
    print("========== BSE FALLBACK CLASSIFICATION START ==========")
    if not MASTER_FILE.exists():
        raise FileNotFoundError(f"Missing master file: {MASTER_FILE}")
    if not UNMAPPED_FILE.exists():
        raise FileNotFoundError(f"Missing BSE exception report: {UNMAPPED_FILE}. Run script 07 first.")

    master = ensure_master_columns(pd.read_parquet(MASTER_FILE))
    report = pd.read_csv(UNMAPPED_FILE, dtype=str).fillna("")
    mapping = load_mapping()

    if "isin" not in report.columns:
        raise ValueError(f"Exception report lacks ISIN column: {UNMAPPED_FILE}")

    report["isin"] = text_series(report["isin"])
    report["symbol"] = text_series(report.get("symbol", pd.Series("", index=report.index)))
    report["company_name"] = text_series(report.get("company_name", pd.Series("", index=report.index)))

    eligible_isins = set(report["isin"]) - {""}
    candidates = master[
        master["isin"].isin(eligible_isins)
        & ~complete_mask(master)
        & text_series(master["classification_status"]).isin(RETRYABLE_STATUSES)
    ].copy()
    candidates = candidates.sort_values(["fallback_bse_attempt_count", "symbol", "isin"]).head(BATCH_SIZE)

    print(f"Exception-report rows: {len(report):,}")
    print(f"Fallback candidates in this run: {len(candidates):,}")

    if candidates.empty:
        still = write_still_unmapped(master)
        master.to_parquet(MASTER_FILE, index=False)
        print(f"No eligible fallback candidates. Still incomplete: {len(still):,}")
        return

    now_text = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    new_mapping_rows: list[dict] = []
    bse = BSE(download_folder=str(DOWNLOAD_FOLDER))

    try:
        for number, (master_index, row) in enumerate(candidates.iterrows(), start=1):
            isin = clean(row["isin"])
            symbol = clean(row["symbol"])
            company_name = clean(row["company_name"])
            prior_count = int(master.at[master_index, "fallback_bse_attempt_count"])
            print(f"[{number}/{len(candidates)}] {symbol} | {isin} | fallback attempts: {prior_count}")

            bse_code, lookup_method, failure_reason = find_bse_record(bse, isin, symbol, company_name)
            metadata: dict = {}
            if bse_code:
                try:
                    candidate_metadata = bse.equityMetaInfo(bse_code) or {}
                    metadata = candidate_metadata if isinstance(candidate_metadata, dict) else {}
                    if not metadata:
                        failure_reason = "BSE metadata response was empty or invalid"
                except Exception as exc:
                    failure_reason = f"equityMetaInfo {type(exc).__name__}: {exc}"

            sector, industry, basic_industry = extract_hierarchy(metadata)
            complete = bool(sector and industry and basic_industry)

            master.at[master_index, "fallback_bse_attempt_count"] = prior_count + 1
            master.at[master_index, "fallback_bse_last_attempt_utc"] = now_text
            master.at[master_index, "fallback_bse_attempted"] = "YES"

            if complete:
                master.at[master_index, "sector"] = sector
                master.at[master_index, "industry"] = industry
                master.at[master_index, "basic_industry"] = basic_industry
                master.at[master_index, "classification_status"] = "CLASSIFIED"
                master.at[master_index, "classification_source"] = "BSE equityMetaInfo fallback"
                master.at[master_index, "classification_failure_reason"] = ""
                status, source, reason = "CLASSIFIED", "BSE equityMetaInfo fallback", ""
                print(f"  Resolved via {lookup_method}: {sector} | {industry} | {basic_industry}")
            else:
                master.at[master_index, "classification_status"] = "BSE_RETRY"
                master.at[master_index, "classification_failure_reason"] = failure_reason or "BSE returned incomplete hierarchy"
                status, source, reason = "BSE_RETRY", "", master.at[master_index, "classification_failure_reason"]
                print(f"  Still retryable: {reason}")

            new_mapping_rows.append({
                "isin": isin,
                "symbol": symbol,
                "company_name": company_name,
                "series": clean(row.get("series")),
                "bse_code": bse_code,
                "bse_sector": sector,
                "bse_industry": clean(metadata.get("Industry")),
                "bse_industry_new": clean(metadata.get("IndustryNew")),
                "bse_i_group": clean(metadata.get("IGroup")),
                "bse_i_sub_group": clean(metadata.get("ISubGroup")),
                "classification_status": status,
                "classification_source": source,
                "failure_reason": reason,
                "attempted_at_utc": now_text,
                "attempt_count": prior_count + 1,
                "bse_lookup_method": lookup_method,
            })
            time.sleep(REQUEST_DELAY_SECONDS)
    finally:
        try:
            bse.exit()
        except Exception:
            pass

    if new_mapping_rows:
        mapping = pd.concat([mapping, pd.DataFrame(new_mapping_rows)], ignore_index=True)
        mapping["attempt_count"] = pd.to_numeric(mapping["attempt_count"], errors="coerce").fillna(0).astype(int)
        mapping = mapping.drop_duplicates("isin", keep="last").sort_values(["symbol", "isin"]).reset_index(drop=True)

    master = master.drop_duplicates("isin", keep="last").sort_values(["symbol", "series", "isin"]).reset_index(drop=True)
    master.to_parquet(MASTER_FILE, index=False)
    mapping.to_csv(MAPPING_FILE, index=False)
    still = write_still_unmapped(master)

    print("========== BSE FALLBACK CLASSIFICATION COMPLETE ==========")
    print(f"Processed this run: {len(new_mapping_rows):,}")
    print(f"Complete hierarchy records: {int(complete_mask(master).sum()):,}")
    print(f"Still incomplete/retryable: {len(still):,}")
    print(f"Updated master: {MASTER_FILE}")
    print(f"Updated mapping audit: {MAPPING_FILE}")
    print(f"Still-unmapped report: {STILL_UNMAPPED_FILE}")


if __name__ == "__main__":
    main()
