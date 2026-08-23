from pathlib import Path
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import urllib.request
import urllib.error


ROOT = Path(__file__).resolve().parents[1]

MASTER_FILE = ROOT / "data" / "processed" / "nse_mainboard_master_bse_classified.parquet"


# NSE bhavcopy archive URL pattern
NSE_BHAVCOPY_BASE = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_"

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "*/*",
}


BSE_SEARCH_URL = "https://api.bseindia.com/bseapi/SecuritySearch"
BSE_DETAILS_URL = "https://api.bseindia.com/bseapi/SecurityDetails"

BSE_API_CODE = "23082025"
BSE_API_KEY = "7271323c40484e188961f1d653c4b923"


YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v10/finance/quoteSummary"


def download_nse_bhavcopy_csv(date_str):
    """
    Download NSE bhavcopy CSV for a given date string (DDMMYYYY).
    Returns raw CSV text or raises an exception.
    """
    file_name = f"sec_bhavdata_full_{date_str}.csv"
    url = NSE_BHAVCOPY_BASE + file_name

    # First, check existence with HEAD
    try:
        head_req = urllib.request.Request(url, headers=NSE_HEADERS, method="HEAD")
        with urllib.request.urlopen(head_req, timeout=15) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HEAD status {resp.status}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"NSE bhavcopy not available for {date_str}: {e.code}")
    except Exception as e:
        raise RuntimeError(f"HEAD request failed for {date_str}: {e}")

    # Now download with GET
    try:
        get_req = urllib.request.Request(url, headers=NSE_HEADERS)
        with urllib.request.urlopen(get_req, timeout=30) as resp:
            csv_bytes = resp.read()
        csv_text = csv_bytes.decode("utf-8", errors="ignore")
        return csv_text
    except Exception as e:
        raise RuntimeError(f"GET request failed for {date_str}: {e}")


def download_nse_mainboard_list():
    """
    Try to download NSE bhavcopy for the last trading day (skip weekends),
    then up to 5 previous days if needed.
    Returns a DataFrame with columns: symbol, series, isin, company_name, listing_date.
    """
    now = datetime.utcnow()
    tried_dates = []

    for offset in range(1, 8):  # try up to 7 days back
        date = now - timedelta(days=offset)
        # Skip weekends
        if date.weekday() >= 5:
            continue

        date_str = date.strftime("%d%m%Y")
        tried_dates.append(date_str)

        try:
            csv_text = download_nse_bhavcopy_csv(date_str)
        except Exception:
            continue

        try:
            df = pd.read_csv(StringIO(csv_text))
        except Exception:
            continue

        # Expected columns: SYMBOL, SERIES, ISIN, COMPANY NAME, ...
        rename_map = {
            "SYMBOL": "symbol",
            "SERIES": "series",
            "ISIN": "isin",
            "COMPANY NAME": "company_name",
        }

        df = df.rename(columns=rename_map)

        required_columns = ["symbol", "series", "isin", "company_name"]
        if not all(col in df.columns for col in required_columns):
            continue

        # Filter to EQ only
        df = df[df["series"] == "EQ"].copy()

        if len(df) == 0:
            continue

        df["listing_date"] = ""  # Not available in bhavcopy

        for column in ["symbol", "company_name", "series", "isin", "listing_date"]:
            df[column] = df[column].fillna("").astype(str).str.strip()

        df = df.drop_duplicates(subset=["symbol", "series"]).reset_index(drop=True)

        if len(df) > 0:
            return df

    raise RuntimeError(
        "Failed to download NSE bhavcopy for recent dates. "
        f"Tried: {', '.join(tried_dates)}"
    )


def clean(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def classify_via_bse_api(symbol, company_name, series, isin):
    import requests

    session = requests.Session()

    search_headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "bse-api-key": BSE_API_KEY,
        "bse-api-code": BSE_API_CODE,
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    search_payload = {
        "AuthCode": BSE_API_CODE,
        "ApiKey": BSE_API_KEY,
        "SearchString": symbol,
        "SecurityType": "EQ",
    }

    try:
        search_response = session.post(
            BSE_SEARCH_URL,
            json=search_payload,
            headers=search_headers,
            timeout=15,
        )
        search_response.raise_for_status()
        search_data = search_response.json()
    except Exception:
        return {
            "status": "FAILED",
            "reason": "BSE search request failed",
            "sector": "",
            "industry": "",
            "basic_industry": "",
        }

    if not isinstance(search_data, dict):
        return {
            "status": "FAILED",
            "reason": "BSE search response invalid",
            "sector": "",
            "industry": "",
            "basic_industry": "",
        }

    body = search_data.get("body", {})
    if not isinstance(body, dict):
        return {
            "status": "FAILED",
            "reason": "BSE search body missing",
            "sector": "",
            "industry": "",
            "basic_industry": "",
        }

    results = body.get("results", [])
    if not isinstance(results, list) or len(results) == 0:
        return {
            "status": "NOT_FOUND",
            "reason": "No matching BSE security",
            "sector": "",
            "industry": "",
            "basic_industry": "",
        }

    matched = None

    for candidate in results:
        if not isinstance(candidate, dict):
            continue

        candidate_symbol = clean(candidate.get("SecurityCode", ""))
        candidate_isin = clean(candidate.get("ISIN", ""))

        if candidate_symbol == symbol and candidate_isin == isin:
            matched = candidate
            break

    if matched is None:
        for candidate in results:
            if not isinstance(candidate, dict):
                continue
            candidate_isin = clean(candidate.get("ISIN", ""))
            if candidate_isin == isin:
                matched = candidate
                break

    if matched is None:
        return {
            "status": "NOT_FOUND",
            "reason": "No exact BSE match",
            "sector": "",
            "industry": "",
            "basic_industry": "",
        }

    security_code = clean(matched.get("SecurityCode", ""))

    details_payload = {
        "AuthCode": BSE_API_CODE,
        "ApiKey": BSE_API_KEY,
        "SecurityCode": security_code,
    }

    try:
        details_response = session.post(
            BSE_DETAILS_URL,
            json=details_payload,
            headers=search_headers,
            timeout=15,
        )
        details_response.raise_for_status()
        details_data = details_response.json()
    except Exception:
        return {
            "status": "FAILED",
            "reason": "BSE details request failed",
            "sector": "",
            "industry": "",
            "basic_industry": "",
        }

    if not isinstance(details_data, dict):
        return {
            "status": "FAILED",
            "reason": "BSE details response invalid",
            "sector": "",
            "industry": "",
            "basic_industry": "",
        }

    details_body = details_data.get("body", {})
    if not isinstance(details_body, dict):
        return {
            "status": "FAILED",
            "reason": "BSE details body missing",
            "sector": "",
            "industry": "",
            "basic_industry": "",
        }

    results_list = details_body.get("results", [])
    if not isinstance(results_list, list) or len(results_list) == 0:
        return {
            "status": "FAILED",
            "reason": "BSE details results missing",
            "sector": "",
            "industry": "",
            "basic_industry": "",
        }

    details = results_list[0]
    if not isinstance(details, dict):
        return {
            "status": "FAILED",
            "reason": "BSE details entry invalid",
            "sector": "",
            "industry": "",
            "basic_industry": "",
        }

    sector = clean(details.get("Sector", ""))
    industry = clean(details.get("Industry", ""))
    basic_industry = clean(details.get("BasicIndustry", ""))

    if not sector and not industry and not basic_industry:
        return {
            "status": "FAILED",
            "reason": "BSE details missing industry fields",
            "sector": "",
            "industry": "",
            "basic_industry": "",
        }

    return {
        "status": "CLASSIFIED",
        "reason": "",
        "sector": sector,
        "industry": industry,
        "basic_industry": basic_industry,
    }


def classify_via_yahoo_finance(symbol, company_name, series, isin):
    import requests

    session = requests.Session()

    yahoo_headers = {
        "Accept": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    search_params = {
        "q": f"{symbol} NSE",
    }

    try:
        search_response = session.get(
            YAHOO_SEARCH_URL,
            params=search_params,
            headers=yahoo_headers,
            timeout=15,
        )
        search_response.raise_for_status()
        search_data = search_response.json()
    except Exception:
        return {
            "status": "FAILED",
            "reason": "Yahoo search request failed",
            "yahoo_ticker": "",
            "yahoo_sector": "",
            "yahoo_industry": "",
        }

    if not isinstance(search_data, dict):
        return {
            "status": "FAILED",
            "reason": "Yahoo search response invalid",
            "yahoo_ticker": "",
            "yahoo_sector": "",
            "yahoo_industry": "",
        }

    quotes = search_data.get("quotes", [])
    if not isinstance(quotes, list) or len(quotes) == 0:
        return {
            "status": "NOT_FOUND",
            "reason": "No Yahoo quote found",
            "yahoo_ticker": "",
            "yahoo_sector": "",
            "yahoo_industry": "",
        }

    matched_quote = None

    for quote in quotes:
        if not isinstance(quote, dict):
            continue
        quote_symbol = clean(quote.get("symbol", ""))
        if quote_symbol == f"{symbol}.NS":
            matched_quote = quote
            break

    if matched_quote is None:
        for quote in quotes:
            if not isinstance(quote, dict):
                continue
            quote_symbol = clean(quote.get("symbol", ""))
            if quote_symbol.endswith(".NS"):
                matched_quote = quote
                break

    if matched_quote is None:
        return {
            "status": "NOT_FOUND",
            "reason": "No suitable Yahoo ticker",
            "yahoo_ticker": "",
            "yahoo_sector": "",
            "yahoo_industry": "",
        }

    yahoo_ticker = clean(matched_quote.get("symbol", ""))

    quote_params = {
        "symbol": yahoo_ticker,
    }

    try:
        quote_response = session.get(
            YAHOO_QUOTE_URL,
            params=quote_params,
            headers=yahoo_headers,
            timeout=15,
        )
        quote_response.raise_for_status()
        quote_data = quote_response.json()
    except Exception:
        return {
            "status": "FAILED",
            "reason": "Yahoo quote request failed",
            "yahoo_ticker": yahoo_ticker,
            "yahoo_sector": "",
            "yahoo_industry": "",
        }

    if not isinstance(quote_data, dict):
        return {
            "status": "FAILED",
            "reason": "Yahoo quote response invalid",
            "yahoo_ticker": yahoo_ticker,
            "yahoo_sector": "",
            "yahoo_industry": "",
        }

    quote_summary = quote_data.get("quoteSummary", {})
    if not isinstance(quote_summary, dict):
        return {
            "status": "FAILED",
            "reason": "Yahoo quoteSummary missing",
            "yahoo_ticker": yahoo_ticker,
            "yahoo_sector": "",
            "yahoo_industry": "",
        }

    asset_profile = quote_summary.get("assetProfile", {})
    if not isinstance(asset_profile, dict):
        return {
            "status": "NOT_FOUND",
            "reason": "Yahoo assetProfile missing",
            "yahoo_ticker": yahoo_ticker,
            "yahoo_sector": "",
            "yahoo_industry": "",
        }

    yahoo_sector = clean(asset_profile.get("sector", ""))
    yahoo_industry = clean(asset_profile.get("industry", ""))

    if not yahoo_sector and not yahoo_industry:
        return {
            "status": "NOT_FOUND",
            "reason": "Yahoo sector/industry missing",
            "yahoo_ticker": yahoo_ticker,
            "yahoo_sector": "",
            "yahoo_industry": "",
        }

    return {
        "status": "YAHOO_FALLBACK",
        "reason": "",
        "yahoo_ticker": yahoo_ticker,
        "yahoo_sector": yahoo_sector,
        "yahoo_industry": yahoo_industry,
    }


def main():
    print("========== NSE UNIVERSE UPDATE START ==========")

    current = pd.read_parquet(MASTER_FILE)

    for column in [
        "classification_status",
        "classification_source",
        "sector",
        "industry",
        "basic_industry",
        "yahoo_ticker",
        "yahoo_sector",
        "yahoo_industry",
        "classification_failure_reason",
        "yahoo_failure_reason",
    ]:
        if column not in current.columns:
            current[column] = ""

    current["classification_status"] = (
        current["classification_status"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    print(f"Current master size: {len(current)}")

    try:
        latest = download_nse_mainboard_list()
    except Exception as error:
        print(f"Failed to download NSE list: {error}")
        return

    print(f"Latest NSE mainboard size: {len(latest)}")

    current_isins = set(current["isin"].dropna().astype(str).str.strip())

    latest["isin"] = (
        latest["isin"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    new_stocks = latest[
        ~latest["isin"].isin(current_isins)
    ].copy()

    if len(new_stocks) == 0:
        print("No new stocks detected. Universe unchanged.")
        return

    print(f"New stocks detected: {len(new_stocks)}")

    for column in [
        "classification_status",
        "classification_source",
        "sector",
        "industry",
        "basic_industry",
        "yahoo_ticker",
        "yahoo_sector",
        "yahoo_industry",
        "classification_failure_reason",
        "yahoo_failure_reason",
    ]:
        new_stocks[column] = ""

    new_stocks["classification_status"] = "PENDING"

    print("Classifying new stocks via BSE...")

    bse_results = []

    for index, row in new_stocks.iterrows():
        symbol = clean(row.get("symbol", ""))
        company_name = clean(row.get("company_name", ""))
        series = clean(row.get("series", ""))
        isin = clean(row["isin"])

        if not symbol or not isin:
            bse_results.append({
                "symbol": symbol,
                "isin": isin,
                "status": "FAILED",
                "reason": "Missing symbol or ISIN",
                "sector": "",
                "industry": "",
                "basic_industry": "",
            })
            continue

        result = classify_via_bse_api(symbol, company_name, series, isin)

        bse_results.append({
            "symbol": symbol,
            "isin": isin,
            "status": result.get("status", "FAILED"),
            "reason": result.get("reason", ""),
            "sector": result.get("sector", ""),
            "industry": result.get("industry", ""),
            "basic_industry": result.get("basic_industry", ""),
        })

    bse_df = pd.DataFrame(bse_results)

    new_stocks = new_stocks.merge(
        bse_df,
        on=["symbol", "isin"],
        how="left",
        suffixes=("", "_bse"),
    )

    classified_mask = new_stocks["status"] == "CLASSIFIED"

    new_stocks.loc[classified_mask, "classification_status"] = "CLASSIFIED"
    new_stocks.loc[classified_mask, "classification_source"] = "BSE"
    new_stocks.loc[classified_mask, "sector"] = new_stocks.loc[
        classified_mask, "sector"
    ]
    new_stocks.loc[classified_mask, "industry"] = new_stocks.loc[
        classified_mask, "industry"
    ]
    new_stocks.loc[classified_mask, "basic_industry"] = new_stocks.loc[
        classified_mask, "basic_industry"
    ]

    pending_mask = new_stocks["classification_status"] == "PENDING"

    print(f"BSE classified: {int(classified_mask.sum())}")
    print(f"Pending Yahoo fallback: {int(pending_mask.sum())}")

    if pending_mask.sum() > 0:
        print("Classifying pending stocks via Yahoo...")

        pending = new_stocks[pending_mask].copy()

        yahoo_results = []

        for index, row in pending.iterrows():
            symbol = clean(row.get("symbol", ""))
            company_name = clean(row.get("company_name", ""))
            series = clean(row.get("series", ""))
            isin = clean(row["isin"])

            if not symbol or not isin:
                yahoo_results.append({
                    "symbol": symbol,
                    "isin": isin,
                    "status": "FAILED",
                    "reason": "Missing symbol or ISIN",
                    "yahoo_ticker": "",
                    "yahoo_sector": "",
                    "yahoo_industry": "",
                })
                continue

            result = classify_via_yahoo_finance(
                symbol, company_name, series, isin
            )

            yahoo_results.append({
                "symbol": symbol,
                "isin": isin,
                "status": result.get("status", "FAILED"),
                "reason": result.get("reason", ""),
                "yahoo_ticker": result.get("yahoo_ticker", ""),
                "yahoo_sector": result.get("yahoo_sector", ""),
                "yahoo_industry": result.get("yahoo_industry", ""),
            })

        yahoo_df = pd.DataFrame(yahoo_results)

        pending = pending.merge(
            yahoo_df,
            on=["symbol", "isin"],
            how="left",
            suffixes=("", "_yahoo"),
        )

        yahoo_classified_mask = pending["status"] == "YAHOO_FALLBACK"

        pending.loc[
            yahoo_classified_mask, "classification_status"
        ] = "YAHOO_FALLBACK"
        pending.loc[
            yahoo_classified_mask, "classification_source"
        ] = "YAHOO"
        pending.loc[
            yahoo_classified_mask, "yahoo_ticker"
        ] = pending.loc[yahoo_classified_mask, "yahoo_ticker"]
        pending.loc[
            yahoo_classified_mask, "yahoo_sector"
        ] = pending.loc[yahoo_classified_mask, "yahoo_sector"]
        pending.loc[
            yahoo_classified_mask, "yahoo_industry"
        ] = pending.loc[yahoo_classified_mask, "yahoo_industry"]

        yahoo_failed_mask = pending["classification_status"] == "PENDING"

        pending.loc[
            yahoo_failed_mask, "classification_status"
        ] = "NOT_FOUND"
        pending.loc[
            yahoo_failed_mask, "classification_source"
        ] = ""

        new_stocks.loc[pending_mask] = pending

        print(f"Yahoo classified: {int(yahoo_classified_mask.sum())}")
        print(f"Still not found: {int(yahoo_failed_mask.sum())}")

    updated = pd.concat(
        [current, new_stocks],
        ignore_index=True,
    )

    updated = updated.sort_values(["symbol", "series"]).reset_index(drop=True)

    updated.to_parquet(MASTER_FILE, index=False)

    print("========== NSE UNIVERSE UPDATE COMPLETE ==========")
    print(f"Updated master size: {len(updated)}")
    print(f"New stocks added: {len(new_stocks)}")
    print(f"Updated master: {MASTER_FILE}")


if __name__ == "__main__":
    main()
