# scripts/01_build_group_features.py
# NSE Sectoral Breadth — 2-Axis Engine v2
# Heavy calculations run in GitHub Actions. Streamlit reads only prepared files.

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import load_settings, p, read_parquet_safe, write_parquet

DEFAULT_VOLUME_SHOCK_THRESHOLD = 1.5
DEFAULT_HIGH_STRENGTH_THRESHOLD = 70.0
DEFAULT_SMALL_GROUP_LIMIT = 5


def add_stock_indicators(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    data = df.sort_values(["symbol", "date"]).copy()
    analysis = settings.get("analysis", {})
    volume_shock_threshold = analysis.get("volume_shock_threshold", DEFAULT_VOLUME_SHOCK_THRESHOLD)
    grouped = data.groupby("symbol", group_keys=False)

    for span in [10, 20, 50, 150, 200]:
        data[f"ema_{span}"] = grouped["close"].transform(
            lambda series: series.ewm(span=span, min_periods=max(5, span // 4)).mean()
        )
        data[f"sma_{span}"] = data[f"ema_{span}"]

    data["avg_vol_20"] = grouped["volume"].transform(lambda series: series.rolling(20, min_periods=10).mean())
    data["avg_vol_50"] = grouped["volume"].transform(lambda series: series.rolling(50, min_periods=15).mean())
    data["avg_val_20"] = grouped["turnover"].transform(lambda series: series.rolling(20, min_periods=10).mean())
    data["ret_1d"] = grouped["close"].pct_change(1)

    for window in analysis.get("return_windows", [5, 10, 20, 60]):
        data[f"ret_{window}d"] = grouped["close"].pct_change(window)

    data["above_20"] = (data["close"] > data["ema_20"]).astype(int)
    data["above_50"] = (data["close"] > data["ema_50"]).astype(int)
    data["above_200"] = (data["close"] > data["ema_200"]).astype(int)
    data["dist_52w_high"] = grouped["close"].transform(lambda series: series / series.rolling(252, min_periods=60).max() - 1.0)
    data["new_high_55"] = (data["close"] >= grouped["high"].transform(lambda series: series.rolling(55, min_periods=20).max())).astype(int)
    data["new_high_252"] = (data["close"] >= grouped["high"].transform(lambda series: series.rolling(252, min_periods=60).max())).astype(int)

    breakout_high = grouped["high"].transform(lambda series: series.shift(1).rolling(analysis.get("breakout_lookback", 55), min_periods=20).max())
    accumulation_multiplier = analysis.get("accumulation_volume_multiplier", 1.2)
    distribution_multiplier = analysis.get("distribution_volume_multiplier", 1.2)
    data["breakout_55"] = ((data["close"] > breakout_high) & (data["volume"] > accumulation_multiplier * data["avg_vol_20"])).astype(int)
    data["acc_day"] = ((data["close"] > grouped["close"].shift(1)) & (data["volume"] > accumulation_multiplier * data["avg_vol_20"])).astype(int)
    data["dist_day"] = ((data["close"] < grouped["close"].shift(1)) & (data["volume"] > distribution_multiplier * data["avg_vol_20"])).astype(int)
    data["volume_shock_ratio"] = np.where(data["avg_vol_20"] > 0, data["volume"] / data["avg_vol_20"], np.nan)
    data["buy_volume_shock"] = ((data["volume_shock_ratio"] >= volume_shock_threshold) & (data["ret_1d"] > 0)).astype(int)
    data["sell_volume_shock"] = ((data["volume_shock_ratio"] >= volume_shock_threshold) & (data["ret_1d"] < 0)).astype(int)
    data["full_alignment"] = ((data["ema_20"] > data["ema_50"]) & (data["ema_50"] > data["ema_200"])).astype(int)
    data["trend_template_pass"] = ((data["close"] > data["ema_50"]) & (data["close"] > data["ema_200"]) & (data["full_alignment"] == 1) & (data["dist_52w_high"] > -0.25)).astype(int)

    data["up_vol"] = np.where(data["ret_1d"] > 0, data["volume"], 0.0)
    data["down_vol"] = np.where(data["ret_1d"] < 0, data["volume"], 0.0)
    up_volume_50 = grouped["up_vol"].transform(lambda series: series.rolling(50, min_periods=15).sum())
    down_volume_50 = grouped["down_vol"].transform(lambda series: series.rolling(50, min_periods=15).sum())
    data["up_down_ratio"] = np.where(down_volume_50 > 0, up_volume_50 / down_volume_50, 1.0)

    previous_close = grouped["close"].shift(1)
    true_range = np.maximum(data["high"] - data["low"], np.maximum((data["high"] - previous_close).abs(), (data["low"] - previous_close).abs()))
    data["atr_14"] = true_range.groupby(data["symbol"]).transform(lambda series: series.rolling(14, min_periods=5).mean())
    high_3 = grouped["high"].transform(lambda series: series.rolling(3, min_periods=3).max())
    low_3 = grouped["low"].transform(lambda series: series.rolling(3, min_periods=3).min())
    data["range_3d"] = high_3 - low_3
    data["tight_3d_range"] = np.where(data["close"] > 0, data["range_3d"] / data["close"], np.nan)
    data["daily_range"] = np.where(data["close"] > 0, (data["high"] - data["low"]) / data["close"], np.nan)

    high_5 = grouped["high"].transform(lambda series: series.rolling(5, min_periods=3).max())
    low_5 = grouped["low"].transform(lambda series: series.rolling(5, min_periods=3).min())
    data["tightness_squeeze_5d"] = (high_5 / low_5) - 1.0
    data["tight_squeeze_pass"] = (data["tightness_squeeze_5d"] <= 0.08).astype(int)
    adr_20 = grouped["daily_range"].transform(lambda series: series.rolling(20, min_periods=10).mean())
    adr_5 = grouped["daily_range"].transform(lambda series: series.rolling(5, min_periods=3).mean())
    data["tight_adr_pass"] = ((adr_5 <= 0.5 * adr_20) & (data["daily_range"] <= 0.05)).astype(int)
    data["tight_consecutive_pass"] = (grouped["daily_range"].transform(lambda series: (series <= 0.05).rolling(3, min_periods=3).min()) == 1).astype(int)
    data["price_tightness_pass"] = (data["range_3d"] <= 1.2 * data["atr_14"]).astype(int)

    high_20 = grouped["high"].transform(lambda series: series.rolling(20, min_periods=10).max())
    low_20 = grouped["low"].transform(lambda series: series.rolling(20, min_periods=10).min())
    data["range_20"] = (high_20 / low_20) - 1.0
    data["vcp_ready"] = ((data["range_20"] <= analysis.get("vcp_range_contraction_threshold", 0.08)) & (data["volume"] <= analysis.get("vcp_volume_dryup_threshold", 0.65) * data["avg_vol_20"]) & (data["dist_52w_high"] > -analysis.get("pivot_proximity_threshold", 0.05))).astype(int)

    low_125 = grouped["low"].transform(lambda series: series.rolling(125, min_periods=25).min())
    data["gain_6m"] = np.where(low_125 > 0, data["close"] / low_125 - 1.0, 0.0)
    data["vol_ratio_50"] = np.where(data["avg_vol_50"] > 0, data["volume"] / data["avg_vol_50"], np.nan)
    vol_2x_hit = (data["volume"] >= 2.0 * data["avg_vol_50"]).astype(int)
    data["vol_2x_count_6m"] = vol_2x_hit.groupby(data["symbol"]).transform(lambda series: series.rolling(125, min_periods=25).sum())
    data["max_vol_ratio_6m"] = grouped["vol_ratio_50"].transform(lambda series: series.rolling(125, min_periods=25).max())
    data["vol_dryup_pass"] = (data["volume"] <= 0.5 * data["avg_vol_50"]).astype(int)

    data["is_new_high_20"] = (data["close"] >= high_20).astype(int)
    data["is_new_low_20"] = (data["close"] <= low_20).astype(int)
    data["nh_nl_val"] = data["is_new_high_20"] - data["is_new_low_20"]

    rule_liquidity = data["avg_val_20"] >= 50_000_000
    rule_trend = ((data["close"] > data["ema_50"]) & (data["ema_20"] > data["ema_50"]) & (data["ema_50"] > data["ema_200"]) & (data["dist_52w_high"] >= -0.25))
    precision_pool_mask = rule_liquidity & rule_trend

    pool_data = data.loc[precision_pool_mask, ["date", "gain_6m", "range_3d", "atr_14", "volume", "avg_vol_50"]].copy()
    data["setup_precision_score"] = np.nan
    if not pool_data.empty:
        pool_data["coil_raw"] = pool_data["range_3d"] / pool_data["atr_14"].clip(lower=1e-9)
        pool_data["dryup_raw"] = pool_data["volume"] / pool_data["avg_vol_50"].clip(lower=1e-9)
        power_points = pool_data.groupby("date")["gain_6m"].rank(pct=True) * 20.0
        coil_points = (1.0 - pool_data.groupby("date")["coil_raw"].rank(pct=True)) * 35.0
        dryup_points = (1.0 - pool_data.groupby("date")["dryup_raw"].rank(pct=True)) * 45.0
        data.loc[pool_data.index, "setup_precision_score"] = (power_points + coil_points + dryup_points).round(1)
    data["actionable_setup_pass"] = (precision_pool_mask & (data["setup_precision_score"] >= 60.0)).astype(int)

    data["nearest_ema_tag"] = "N/A"
    valid_ema = data[["ema_10", "ema_20", "ema_50"]].notna().any(axis=1)
    if valid_ema.any():
        subset = data.loc[valid_ema, ["close", "ema_10", "ema_20", "ema_50"]].copy()
        distances = pd.DataFrame({
            "EMA 10": (subset["close"] - subset["ema_10"].clip(lower=1e-9)) / subset["ema_10"].clip(lower=1e-9),
            "EMA 20": (subset["close"] - subset["ema_20"].clip(lower=1e-9)) / subset["ema_20"].clip(lower=1e-9),
            "EMA 50": (subset["close"] - subset["ema_50"].clip(lower=1e-9)) / subset["ema_50"].clip(lower=1e-9),
        })
        nearest_name = distances.abs().idxmin(axis=1)
        # DataFrame.lookup was removed from pandas. Select row-specific values
        # by reindexing the chosen column labels, which is pandas 2.x-safe.
        nearest_distance = pd.Series(
            [distances.at[index, column] for index, column in nearest_name.items()],
            index=distances.index,
            dtype=float,
        )

        def ema_tag(distance: float) -> str:
            if pd.isna(distance): return "N/A"
            if -0.01 <= distance <= 0.01: return "On EMA"
            if 0.01 < distance <= 0.05: return f"Riding +{distance * 100:.0f}%"
            if distance > 0.05: return f"Extended +{distance * 100:.0f}%"
            if -0.05 <= distance < -0.01: return f"Testing -{abs(distance) * 100:.0f}%"
            return f"Broken -{abs(distance) * 100:.0f}%"

        data.loc[valid_ema, "nearest_ema_tag"] = nearest_distance.map(ema_tag)

    data["momentum_badge"] = ""
    if not pool_data.empty:
        pool_gain = data.loc[pool_data.index, ["date", "gain_6m"]].copy()
        threshold_75 = pool_gain.groupby("date")["gain_6m"].transform(lambda series: series.quantile(0.75))
        data.loc[pool_data.index, "momentum_badge"] = np.where(pool_gain["gain_6m"] >= threshold_75, "High Momentum", "")

    history_count = grouped["close"].transform(lambda series: series.rolling(200, min_periods=1).count())
    data["is_ipo"] = (history_count < 150).astype(int)
    data["days_listed"] = grouped.cumcount() + 1
    data["turnover_ex_list"] = np.where(data["days_listed"] == 1, np.nan, data["turnover"])
    expanding_turnover = grouped["turnover_ex_list"].transform(lambda series: series.expanding().mean())
    rolling_turnover = grouped["turnover_ex_list"].transform(lambda series: series.rolling(20, min_periods=1).mean())
    data["ipo_turnover_avg"] = np.where(data["days_listed"] < 21, expanding_turnover, rolling_turnover)
    data["ipo_vol_pass"] = (data["ipo_turnover_avg"] >= 50_000_000).astype(int)
    data["ipo_phase"] = np.select([data["days_listed"] <= 15, data["days_listed"] <= 40], ["discovery", "basing"], default="graduating")
    data["vwap_since_listing"] = grouped["turnover"].transform(lambda series: series.cumsum()) / grouped["volume"].transform(lambda series: series.cumsum()).clip(lower=1e-9)
    data["vwap_premium"] = data["close"] / data["vwap_since_listing"] - 1.0
    high_since_listing = grouped["high"].transform(lambda series: series.cummax())
    data["retracement_from_listing_high"] = data["close"] / high_since_listing - 1.0
    higher_high_low = ((data["high"] > grouped["high"].shift(1)) & (data["low"] > grouped["low"].shift(1))).astype(int)
    data["hh_hl_streak_5d"] = higher_high_low.groupby(data["symbol"]).transform(lambda series: series.rolling(5, min_periods=3).sum())
    range_average_10 = grouped["daily_range"].transform(lambda series: series.rolling(10, min_periods=3).mean())
    data["ipo_tight_pass"] = (data["daily_range"] <= 0.7 * range_average_10).astype(int)

    ipo_mask = data["is_ipo"].eq(1)
    data["ipo_setup_score"] = np.nan
    ipo_data = data.loc[ipo_mask, ["date", "daily_range", "vol_ratio_50", "vwap_premium", "retracement_from_listing_high", "hh_hl_streak_5d"]].copy()
    if not ipo_data.empty:
        tight_points = (1.0 - ipo_data.groupby("date")["daily_range"].rank(pct=True)) * 25.0
        dryup_points = (1.0 - ipo_data.groupby("date")["vol_ratio_50"].rank(pct=True)) * 20.0
        vwap_points = ipo_data.groupby("date")["vwap_premium"].rank(pct=True) * 20.0
        retracement_points = ipo_data.groupby("date")["retracement_from_listing_high"].rank(pct=True) * 20.0
        hh_hl_points = ipo_data.groupby("date")["hh_hl_streak_5d"].rank(pct=True) * 15.0
        data.loc[ipo_data.index, "ipo_setup_score"] = (tight_points.fillna(0) + dryup_points.fillna(0) + vwap_points.fillna(0) + retracement_points.fillna(0) + hh_hl_points.fillna(0)).round(1)

    data["established_buy_setup"] = ((data["is_ipo"] == 0) & (data["actionable_setup_pass"] == 1)).astype(int)
    data["ipo_buy_setup"] = ((data["is_ipo"] == 1) & (data["ipo_vol_pass"] == 1) & (data["ipo_setup_score"] >= 60.0)).astype(int)
    return data


def add_stock_strength(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    data = df.copy()
    threshold = settings.get("analysis", {}).get("high_strength_threshold", DEFAULT_HIGH_STRENGTH_THRESHOLD)
    data["stock_strength_score"] = (data.groupby("date")["ret_20d"].rank(pct=True) * 35.0 + data.groupby("date")["ret_60d"].rank(pct=True) * 35.0 + data["above_50"] * 15.0 + data["above_200"] * 10.0 + data["breakout_55"] * 3.0 + data["vcp_ready"] * 2.0)
    data["high_strength_flag"] = (data["stock_strength_score"] >= threshold).astype(int)
    gain_points = data.groupby("date")["gain_6m"].rank(pct=True) * 35.0
    volume_points = data.groupby("date")["max_vol_ratio_6m"].rank(pct=True) * 35.0
    price_tight_points = (1.0 - data.groupby("date")["tight_3d_range"].rank(pct=True)) * 15.0
    volume_tight_points = (1.0 - data.groupby("date")["vol_ratio_50"].rank(pct=True)) * 15.0
    data["buy_setup_score"] = (gain_points.fillna(0) + volume_points.fillna(0) + price_tight_points.fillna(0) + volume_tight_points.fillna(0)).round(2)
    return data


def aggregate_group(df: pd.DataFrame, group_column: str, settings: dict) -> pd.DataFrame:
    small_group_limit = settings.get("analysis", {}).get("small_industry_limit", DEFAULT_SMALL_GROUP_LIMIT)
    grouped = df.groupby(["date", group_column], dropna=False).agg(
        members=("symbol", "nunique"), eq_ret_1d=("ret_1d", "mean"), eq_ret_5d=("ret_5d", "mean"), eq_ret_10d=("ret_10d", "mean"), eq_ret_20d=("ret_20d", "mean"), eq_ret_60d=("ret_60d", "mean"), med_ret_20d=("ret_20d", "median"), med_ret_60d=("ret_60d", "median"), pct_aligned=("full_alignment", "mean"), med_up_down_ratio=("up_down_ratio", "median"), actionability_raw=("actionable_setup_pass", "mean"), nh_nl_net=("nh_nl_val", "mean"), pct_above_20=("above_20", "mean"), pct_above_50=("above_50", "mean"), pct_above_200=("above_200", "mean"), trend_template_pct=("trend_template_pass", "mean"), new_high_55_pct=("new_high_55", "mean"), new_high_252_pct=("new_high_252", "mean"), acc_days=("acc_day", "sum"), dist_days=("dist_day", "sum"), breakout_count=("breakout_55", "sum"), vcp_ready_count=("vcp_ready", "sum"), high_strength_count=("high_strength_flag", "sum"), buy_volume_shock_count=("buy_volume_shock", "sum"), sell_volume_shock_count=("sell_volume_shock", "sum"), median_volume_shock=("volume_shock_ratio", "median"), median_dist_52w_high=("dist_52w_high", "median"),
    ).reset_index()
    grouped["acc_minus_dist"] = grouped["acc_days"] - grouped["dist_days"]
    for count_column, percent_column in [("breakout_count", "breakout_pct"), ("vcp_ready_count", "vcp_ready_pct"), ("high_strength_count", "pct_high_strength"), ("buy_volume_shock_count", "buy_volume_shock_pct"), ("sell_volume_shock_count", "sell_volume_shock_pct")]:
        grouped[percent_column] = np.where(grouped["members"] > 0, grouped[count_column] / grouped["members"] * 100.0, np.nan)
    grouped["small_industry"] = (grouped["members"] < small_group_limit).astype(int)
    for column in ["pct_above_20", "pct_above_50", "pct_above_200", "trend_template_pct", "new_high_55_pct", "new_high_252_pct"]:
        grouped[column] *= 100.0
    return grouped


def add_group_scores(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    data = df.copy()
    group_column = data.columns[1]
    velocity_20 = data.groupby("date")["med_ret_20d"].rank(pct=True)
    velocity_60 = data.groupby("date")["med_ret_60d"].rank(pct=True)
    velocity_points = ((velocity_20 + velocity_60) / 2.0) * 35.0
    structure_points = data.groupby("date")["pct_aligned"].rank(pct=True) * 35.0
    volume_points = data.groupby("date")["med_up_down_ratio"].rank(pct=True) * 30.0
    raw_score = (velocity_points + structure_points + volume_points).clip(lower=0, upper=100)
    data["leadership_score"] = raw_score.groupby(data[group_column]).transform(lambda series: series.ewm(span=3, min_periods=1).mean()).round(1)
    data["leadership_change_5d"] = data.groupby(group_column)["leadership_score"].diff(5).fillna(0.0).round(1)
    data["improver_priority"] = (0.65 * data["leadership_score"] + 0.35 * data["leadership_change_5d"].clip(lower=0)).round(1)
    data["strength_score"] = data["leadership_score"]
    data["actionability_score"] = (data["actionability_raw"] * 100.0).round(1)
    data["nh_nl_net"] = (data["nh_nl_net"] * 100.0).round(1)
    conditions = [(data["leadership_score"] >= 70) & (data["actionability_score"] >= 15), (data["leadership_score"] >= 70) & (data["actionability_score"] < 15), (data["leadership_score"] < 50) & (data["actionability_score"] >= 15), (data["leadership_score"] < 50) & (data["actionability_score"] < 15)]
    labels = ["Fresh Leader (HUNT)", "Extended Leader (WAIT)", "Speculative Coil (AVOID)", "Dead (AVOID)"]
    data["regime"] = np.select(conditions, labels, default="Neutral Transition")
    return data


def main() -> None:
    settings = load_settings()
    processed = p("data", "processed")
    master = read_parquet_safe(processed / "nse_mainboard_master_bse_classified.parquet")
    prices = read_parquet_safe(processed / "prices.parquet")
    required_prices = ["symbol", "date", "open", "high", "low", "close", "volume"]
    missing_prices = [column for column in required_prices if column not in prices.columns]
    if missing_prices: raise ValueError(f"Prices file missing columns: {missing_prices}")
    if "turnover" not in prices.columns: prices["turnover"] = prices["close"] * prices["volume"]
    required_master = ["symbol", "isin", "industry", "basic_industry", "sector", "series"]
    missing_master = [column for column in required_master if column not in master.columns]
    if missing_master: raise ValueError(f"Master file missing columns: {missing_master}")
    master_for_join = master[required_master + (["mcap"] if "mcap" in master.columns else [])].drop_duplicates("symbol").copy()
    stock = prices.merge(master_for_join, on="symbol", how="left")
    stock["date"] = pd.to_datetime(stock["date"], errors="coerce").dt.normalize()
    stock["series"] = stock["series"].fillna("").astype(str).str.strip()
    stock = stock[stock["series"].eq("EQ")].copy()
    for column in ["sector", "industry", "basic_industry"]:
        stock[column] = stock[column].fillna("Unclassified").astype(str).str.strip().replace("", "Unclassified")
    print(f"Using classified master: {len(master):,} records")
    print(f"EQ price rows after master join: {len(stock):,}")
    print(f"Unclassified stock rows: {int(stock['basic_industry'].eq('Unclassified').sum()):,}")
    stock = add_stock_indicators(stock, settings)
    stock = add_stock_strength(stock, settings)
    write_parquet(stock, processed / "stock_daily_features.parquet")
    write_parquet(add_group_scores(aggregate_group(stock, "industry", settings), settings), processed / "industry_daily_features.parquet")
    write_parquet(add_group_scores(aggregate_group(stock, "basic_industry", settings), settings), processed / "basic_industry_daily_features.parquet")
    write_parquet(add_group_scores(aggregate_group(stock, "sector", settings), settings), processed / "sector_daily_features.parquet")
    print("feature build complete: stock, Basic Industry, Industry and Sector features written")


if __name__ == "__main__":
    main()
