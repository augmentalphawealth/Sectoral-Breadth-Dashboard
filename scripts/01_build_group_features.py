# scripts/01_build_group_features.py
# NSE Sectoral Breadth — 2-Axis Engine v2

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import load_settings, p, read_parquet_safe, write_parquet

DEFAULT_VOLUME_SHOCK_THRESHOLD = 1.5
DEFAULT_HIGH_STRENGTH_THRESHOLD = 70.0
DEFAULT_SMALL_GROUP_LIMIT = 5


def add_stock_indicators(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).copy()
    analysis = settings.get("analysis", {})
    volume_shock_threshold = analysis.get("volume_shock_threshold", DEFAULT_VOLUME_SHOCK_THRESHOLD)
    g = df.groupby("symbol", group_keys=False)

    for span in [10, 20, 50, 150, 200]:
        df[f"ema_{span}"] = g["close"].transform(
            lambda s: s.ewm(span=span, min_periods=max(5, span // 4)).mean()
        )
        df[f"sma_{span}"] = df[f"ema_{span}"]

    df["avg_vol_20"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["avg_vol_50"] = g["volume"].transform(lambda s: s.rolling(50, min_periods=15).mean())
    df["avg_val_20"] = g["turnover"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["ret_1d"] = g["close"].pct_change(1)

    for win in analysis.get("return_windows", [5, 10, 20, 60]):
        df[f"ret_{win}d"] = g["close"].pct_change(win)

    df["above_20"] = (df["close"] > df["ema_20"]).astype(int)
    df["above_50"] = (df["close"] > df["ema_50"]).astype(int)
    df["above_200"] = (df["close"] > df["ema_200"]).astype(int)
    df["dist_52w_high"] = g["close"].transform(lambda s: s / s.rolling(252, min_periods=60).max() - 1)
    df["new_high_55"] = (df["close"] >= g["high"].transform(lambda s: s.rolling(55, min_periods=20).max())).astype(int)
    df["new_high_252"] = (df["close"] >= g["high"].transform(lambda s: s.rolling(252, min_periods=60).max())).astype(int)

    df["breakout_55"] = (
        (df["close"] > g["high"].transform(lambda s: s.shift(1).rolling(analysis.get("breakout_lookback", 55), min_periods=20).max()))
        & (df["volume"] > analysis.get("accumulation_volume_multiplier", 1.2) * df["avg_vol_20"])
    ).astype(int)

    df["acc_day"] = ((df["close"] > g["close"].shift(1)) & (df["volume"] > analysis.get("accumulation_volume_multiplier", 1.2) * df["avg_vol_20"])).astype(int)
    df["dist_day"] = ((df["close"] < g["close"].shift(1)) & (df["volume"] > analysis.get("distribution_volume_multiplier", 1.2) * df["avg_vol_20"])).astype(int)
    df["volume_shock_ratio"] = np.where(df["avg_vol_20"] > 0, df["volume"] / df["avg_vol_20"], np.nan)
    df["buy_volume_shock"] = ((df["volume_shock_ratio"] >= volume_shock_threshold) & (df["ret_1d"] > 0)).astype(int)
    df["sell_volume_shock"] = ((df["volume_shock_ratio"] >= volume_shock_threshold) & (df["ret_1d"] < 0)).astype(int)
    df["full_alignment"] = ((df["ema_20"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"])).astype(int)
    df["trend_template_pass"] = ((df["close"] > df["ema_50"]) & (df["close"] > df["ema_200"]) & (df["full_alignment"] == 1) & (df["dist_52w_high"] > -0.25)).astype(int)

    df["up_vol"] = np.where(df["ret_1d"] > 0, df["volume"], 0.0)
    df["down_vol"] = np.where(df["ret_1d"] < 0, df["volume"], 0.0)
    up_vol_50 = g["up_vol"].transform(lambda s: s.rolling(50, min_periods=15).sum())
    down_vol_50 = g["down_vol"].transform(lambda s: s.rolling(50, min_periods=15).sum())
    df["up_down_ratio"] = np.where(down_vol_50 > 0, up_vol_50 / down_vol_50, 1.0)

    prev_close = g["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"], np.maximum(abs(df["high"] - prev_close), abs(df["low"] - prev_close)))
    df["atr_14"] = tr.groupby(df["symbol"]).transform(lambda s: s.rolling(14, min_periods=5).mean())
    roll_high_3 = g["high"].transform(lambda s: s.rolling(3, min_periods=3).max())
    roll_low_3 = g["low"].transform(lambda s: s.rolling(3, min_periods=3).min())
    df["range_3d"] = roll_high_3 - roll_low_3
    df["tight_3d_range"] = np.where(df["close"] > 0, df["range_3d"] / df["close"], np.nan)
    df["daily_range"] = np.where(df["close"] > 0, (df["high"] - df["low"]) / df["close"], np.nan)

    roll_high_5 = g["high"].transform(lambda s: s.rolling(5, min_periods=3).max())
    roll_low_5 = g["low"].transform(lambda s: s.rolling(5, min_periods=3).min())
    df["tightness_squeeze_5d"] = (roll_high_5 / roll_low_5) - 1.0
    df["tight_squeeze_pass"] = (df["tightness_squeeze_5d"] <= 0.08).astype(int)
    adr_20 = g["daily_range"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    adr_5 = g["daily_range"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    df["tight_adr_pass"] = ((adr_5 <= 0.5 * adr_20) & (df["daily_range"] <= 0.05)).astype(int)
    df["tight_consecutive_pass"] = (g["daily_range"].transform(lambda s: (s <= 0.05).rolling(3, min_periods=3).min()) == 1).astype(int)
    df["price_tightness_pass"] = (df["range_3d"] <= (1.2 * df["atr_14"])).astype(int)

    df["range_20"] = g.apply(lambda x: (x["high"].rolling(20, min_periods=10).max() / x["low"].rolling(20, min_periods=10).min()) - 1).reset_index(level=0, drop=True)
    df["vcp_ready"] = (
        (df["range_20"] <= analysis.get("vcp_range_contraction_threshold", 0.08))
        & (df["volume"] <= analysis.get("vcp_volume_dryup_threshold", 0.65) * df["avg_vol_20"])
        & (df["dist_52w_high"] > -analysis.get("pivot_proximity_threshold", 0.05))
    ).astype(int)

    roll_min_125 = g["low"].transform(lambda s: s.rolling(125, min_periods=25).min())
    df["gain_6m"] = np.where(roll_min_125 > 0, (df["close"] / roll_min_125) - 1.0, 0.0)
    df["vol_ratio_50"] = np.where(df["avg_vol_50"] > 0, df["volume"] / df["avg_vol_50"], np.nan)
    vol_2x_hit = (df["volume"] >= 2.0 * df["avg_vol_50"]).astype(int)
    df["vol_2x_count_6m"] = g["volume"].transform(lambda s: vol_2x_hit.loc[s.index].rolling(125, min_periods=25).sum())
    df["max_vol_ratio_6m"] = g["vol_ratio_50"].transform(lambda s: s.rolling(125, min_periods=25).max())
    df["vol_dryup_pass"] = (df["volume"] <= 0.5 * df["avg_vol_50"]).astype(int)

    high_20 = g["high"].transform(lambda s: s.rolling(20, min_periods=10).max())
    low_20 = g["low"].transform(lambda s: s.rolling(20, min_periods=10).min())
    df["is_new_high_20"] = (df["close"] >= high_20).astype(int)
    df["is_new_low_20"] = (df["close"] <= low_20).astype(int)
    df["nh_nl_val"] = df["is_new_high_20"] - df["is_new_low_20"]

    # ACTIONABLE SETUP
    rule_liquidity = df["avg_val_20"] >= 50000000
    rule_trend = ((df["close"] > df["ema_50"]) & (df["ema_20"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"]) & (df["dist_52w_high"] >= -0.25))
    precision_pool_mask = rule_liquidity & rule_trend
    pool_idx = df.index[precision_pool_mask]

    if len(pool_idx) > 0:
        pool_data = df.loc[pool_idx, ["date", "gain_6m", "range_3d", "atr_14", "volume", "avg_vol_50"]].copy()
        pool_data["coil_raw"] = pool_data["range_3d"] / pool_data["atr_14"].clip(lower=1e-9)
        pool_data["dryup_raw"] = pool_data["volume"] / pool_data["avg_vol_50"].clip(lower=1e-9)
        power_pts = pool_data.groupby("date")["gain_6m"].rank(pct=True, ascending=True) * 20.0
        coil_pts = (1.0 - pool_data.groupby("date")["coil_raw"].rank(pct=True, ascending=True)) * 35.0
        dryup_pts = (1.0 - pool_data.groupby("date")["dryup_raw"].rank(pct=True, ascending=True)) * 45.0
        setup_precision_score = (power_pts + coil_pts + dryup_pts).round(1)
        df.loc[pool_idx, "setup_precision_score"] = setup_precision_score
        df["actionable_setup_pass"] = (precision_pool_mask & (df["setup_precision_score"] >= 60)).astype(int)
    else:
        df["setup_precision_score"] = np.nan
        df["actionable_setup_pass"] = 0

    # EMA PROXIMITY TAG - ROBUST VERSION (no idxmin on all-NA)
    df["nearest_ema_tag"] = "N/A"
    valid_ema_mask = df[["ema_10", "ema_20", "ema_50"]].notna().any(axis=1)
    
    if valid_ema_mask.any():
        ema10 = df.loc[valid_ema_mask, "ema_10"].clip(lower=1e-9)
        ema20 = df.loc[valid_ema_mask, "ema_20"].clip(lower=1e-9)
        ema50 = df.loc[valid_ema_mask, "ema_50"].clip(lower=1e-9)
        close_valid = df.loc[valid_ema_mask, "close"]
        
        dist_10 = (close_valid - ema10) / ema10
        dist_20 = (close_valid - ema20) / ema20
        dist_50 = (close_valid - ema50) / ema50
        
        abs_dist = pd.concat([dist_10.abs(), dist_20.abs(), dist_50.abs()], axis=1)
        nearest_idx = abs_dist.idxmin(axis=1)
        nearest_dist = pd.concat([dist_10, dist_20, dist_50], axis=1).apply(
            lambda row: row[nearest_idx[row.name]], axis=1
        )
        
        def _ema_tag(d):
            if pd.isna(d): return "N/A"
            elif -0.01 <= d <= 0.01: return "On EMA"
            elif 0.01 < d <= 0.05: return f"Riding +{d*100:.0f}%"
            elif d > 0.05: return f"Extended +{d*100:.0f}%"
            elif -0.05 <= d < -0.01: return f"Testing -{abs(d)*100:.0f}%"
            else: return f"Broken -{abs(d)*100:.0f}%"
        
        df.loc[valid_ema_mask, "nearest_ema_tag"] = nearest_dist.apply(_ema_tag)

    # MOMENTUM BADGE
    df["momentum_badge"] = ""
    if len(pool_idx) > 0:
        pool_gain = df.loc[pool_idx, ["date", "gain_6m"]].copy()
        q75 = pool_gain.groupby("date")["gain_6m"].transform(lambda s: s.quantile(0.75))
        df.loc[pool_idx, "momentum_badge"] = np.where(pool_gain["gain_6m"] >= q75, "🔥 High Momentum", "")

    # IPO
    history_count = g["close"].transform(lambda s: s.rolling(200, min_periods=1).count())
    df["is_ipo"] = (history_count < 150).astype(int)
    df["days_listed"] = g.cumcount() + 1
    if "turnover" not in df.columns:
        df["turnover"] = df["close"] * df["volume"]
    df["turnover_ex_list"] = np.where(df["days_listed"] == 1, np.nan, df["turnover"])
    expanding_avg = g["turnover_ex_list"].transform(lambda s: s.expanding().mean())
    rolling_avg = g["turnover_ex_list"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    df["ipo_turnover_avg"] = np.where(df["days_listed"] < 21, expanding_avg, rolling_avg)
    df["ipo_vol_pass"] = (df["ipo_turnover_avg"] >= 50000000).astype(int)
    df["ipo_phase"] = np.select([df["days_listed"] <= 15, df["days_listed"] <= 40], ["discovery", "basing"], default="graduating")
    df["vwap_since_listing"] = g["turnover"].transform(lambda s: s.cumsum()) / g["volume"].transform(lambda s: s.cumsum())
    df["vwap_premium"] = (df["close"] / df["vwap_since_listing"]) - 1.0
    high_since_listing = g["high"].transform(lambda s: s.cummax())
    df["retracement_from_listing_high"] = (df["close"] / high_since_listing) - 1.0
    higher_high = df["high"] > g["high"].shift(1)
    higher_low = df["low"] > g["low"].shift(1)
    hh_hl_day = (higher_high & higher_low).astype(int)
    df["hh_hl_streak_5d"] = g["high"].transform(lambda s: hh_hl_day.loc[s.index].rolling(5, min_periods=3).sum())
    range_avg_10 = g["daily_range"].transform(lambda s: s.rolling(10, min_periods=3).mean())
    df["ipo_tight_pass"] = (df["daily_range"] <= 0.7 * range_avg_10).astype(int)
    ipo_mask = df["is_ipo"] == 1
    ipo_scratch = df.loc[ipo_mask, ["date", "daily_range", "vol_ratio_50", "vwap_premium", "retracement_from_listing_high", "hh_hl_streak_5d"]]
    tight_pts = (1.0 - ipo_scratch.groupby("date")["daily_range"].rank(pct=True)) * 25.0
    dryup_pts = (1.0 - ipo_scratch.groupby("date")["vol_ratio_50"].rank(pct=True)) * 20.0
    vwap_pts = ipo_scratch.groupby("date")["vwap_premium"].rank(pct=True) * 20.0
    retr_pts = ipo_scratch.groupby("date")["retracement_from_listing_high"].rank(pct=True) * 20.0
    hhhl_pts = ipo_scratch.groupby("date")["hh_hl_streak_5d"].rank(pct=True) * 15.0
    df["ipo_setup_score"] = np.nan
    df.loc[ipo_mask, "ipo_setup_score"] = (tight_pts.fillna(0) + dryup_pts.fillna(0) + vwap_pts.fillna(0) + retr_pts.fillna(0) + hhhl_pts.fillna(0)).round(1)
    df["established_buy_setup"] = ((df["is_ipo"] == 0) & (df["actionable_setup_pass"] == 1)).astype(int)
    df["ipo_buy_setup"] = ((df["is_ipo"] == 1) & (df["ipo_vol_pass"] == 1) & (df["ipo_setup_score"] >= 60)).astype(int)

    return df


def add_stock_strength(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    data = df.copy()
    threshold = settings.get("analysis", {}).get("high_strength_threshold", DEFAULT_HIGH_STRENGTH_THRESHOLD)
    data["stock_strength_score"] = (data.groupby("date")["ret_20d"].rank(pct=True) * 35 + data.groupby("date")["ret_60d"].rank(pct=True) * 35 + data["above_50"] * 15 + data["above_200"] * 10 + data["breakout_55"] * 3 + data["vcp_ready"] * 2)
    data["high_strength_flag"] = (data["stock_strength_score"] >= threshold).astype(int)
    gain_pts = data.groupby("date")["gain_6m"].rank(pct=True) * 35
    vol_pts = data.groupby("date")["max_vol_ratio_6m"].rank(pct=True) * 35
    price_tight_pts = (1.0 - data.groupby("date")["tight_3d_range"].rank(pct=True)) * 15
    vol_tight_pts = (1.0 - data.groupby("date")["vol_ratio_50"].rank(pct=True)) * 15
    data["buy_setup_score"] = (gain_pts.fillna(0) + vol_pts.fillna(0) + price_tight_pts.fillna(0) + vol_tight_pts.fillna(0)).round(2)
    return data


def aggregate_group(df: pd.DataFrame, group_col: str, settings: dict) -> pd.DataFrame:
    small_group_limit = settings.get("analysis", {}).get("small_industry_limit", DEFAULT_SMALL_GROUP_LIMIT)
    agg = df.groupby(["date", group_col], dropna=False).agg(
        members=("symbol", "nunique"), eq_ret_1d=("ret_1d", "mean"), eq_ret_5d=("ret_5d", "mean"), eq_ret_10d=("ret_10d", "mean"),
        eq_ret_20d=("ret_20d", "mean"), eq_ret_60d=("ret_60d", "mean"), med_ret_20d=("ret_20d", "median"), med_ret_60d=("ret_60d", "median"),
        pct_aligned=("full_alignment", "mean"), med_up_down_ratio=("up_down_ratio", "median"), actionability_raw=("actionable_setup_pass", "mean"),
        nh_nl_net=("nh_nl_val", "mean"), pct_above_20=("above_20", "mean"), pct_above_50=("above_50", "mean"), pct_above_200=("above_200", "mean"),
        trend_template_pct=("trend_template_pass", "mean"), new_high_55_pct=("new_high_55", "mean"), new_high_252_pct=("new_high_252", "mean"),
        acc_days=("acc_day", "sum"), dist_days=("dist_day", "sum"), breakout_count=("breakout_55", "sum"), vcp_ready_count=("vcp_ready", "sum"),
        high_strength_count=("high_strength_flag", "sum"), buy_volume_shock_count=("buy_volume_shock", "sum"), sell_volume_shock_count=("sell_volume_shock", "sum"),
        median_volume_shock=("volume_shock_ratio", "median"), median_dist_52w_high=("dist_52w_high", "median"),
    ).reset_index()
    agg["acc_minus_dist"] = agg["acc_days"] - agg["dist_days"]
    for count_column, pct_column in [("breakout_count", "breakout_pct"), ("vcp_ready_count", "vcp_ready_pct"), ("high_strength_count", "pct_high_strength"), ("buy_volume_shock_count", "buy_volume_shock_pct"), ("sell_volume_shock_count", "sell_volume_shock_pct")]:
        agg[pct_column] = np.where(agg["members"] > 0, agg[count_column] / agg["members"] * 100, np.nan)
    agg["small_industry"] = (agg["members"] < small_group_limit).astype(int)
    agg["pct_above_20"] *= 100
    agg["pct_above_50"] *= 100
    agg["pct_above_200"] *= 100
    agg["trend_template_pct"] *= 100
    agg["new_high_55_pct"] *= 100
    agg["new_high_252_pct"] *= 100
    return agg


def add_group_scores(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    data = df.copy()
    group_col = data.columns[1]
    v_20 = data.groupby("date")["med_ret_20d"].rank(pct=True)
    v_60 = data.groupby("date")["med_ret_60d"].rank(pct=True)
    velocity_pts = ((v_20 + v_60) / 2.0) * 35.0
    structure_pts = data.groupby("date")["pct_aligned"].rank(pct=True) * 35.0
    volume_pts = data.groupby("date")["med_up_down_ratio"].rank(pct=True) * 30.0
    data["leadership_score"] = (velocity_pts + structure_pts + volume_pts).clip(lower=0, upper=100)
    data["leadership_score"] = data.groupby(group_col)["leadership_score"].transform(lambda s: s.ewm(span=3, min_periods=1).mean()).round(1)
    data["strength_score"] = data["leadership_score"]
    data["actionability_score"] = (data["actionability_raw"] * 100).round(1)
    data["nh_nl_net"] = (data["nh_nl_net"] * 100).round(1)
    conditions = [(data["leadership_score"] >= 70) & (data["actionability_score"] >= 15), (data["leadership_score"] >= 70) & (data["actionability_score"] < 15), (data["leadership_score"] < 50) & (data["actionability_score"] >= 15), (data["leadership_score"] < 50) & (data["actionability_score"] < 15)]
    labels = ["Fresh Leader (HUNT)", "Extended Leader (WAIT)", "Speculative Coil (AVOID)", "Dead (AVOID)"]
    data["regime"] = np.select(conditions, labels, default="Neutral Transition")
    return data


def main() -> None:
    settings = load_settings()
    processed = p("data", "processed")
    master_file = processed / "nse_mainboard_master_bse_classified.parquet"
    prices_file = processed / "prices.parquet"
    master = read_parquet_safe(master_file)
    prices = read_parquet_safe(prices_file)
    print(f"Using classified master: {master_file}")
    print(f"Master stocks: {len(master)}")
    print(f"Price rows: {len(prices)}")
    required_price_columns = ["symbol", "date", "open", "high", "low", "close", "volume"]
    missing_price_columns = [c for c in required_price_columns if c not in prices.columns]
    if missing_price_columns:
        raise ValueError(f"Prices file missing columns: {missing_price_columns}")
    if "turnover" not in prices.columns:
        prices["turnover"] = prices["close"] * prices["volume"]
    required_master_columns = ["symbol", "isin", "industry", "basic_industry", "sector", "series"]
    missing_master_columns = [c for c in required_master_columns if c not in master.columns]
    if missing_master_columns:
        raise ValueError(f"Master missing columns: {missing_master_columns}")
    join_columns = required_master_columns.copy()
    if "mcap" in master.columns:
        join_columns.append("mcap")
    master_for_join = master[join_columns].drop_duplicates(subset=["symbol"]).copy()
    stock = prices.merge(master_for_join, on="symbol", how="left")
    stock["date"] = pd.to_datetime(stock["date"])
    if "mcap" not in stock.columns:
        stock["mcap"] = np.nan
    stock["series"] = stock["series"].fillna("").astype(str).str.strip()
    stock = stock[stock["series"] == "EQ"].copy()
    missing_classification = stock["industry"].isna().sum()
    print(f"Price rows without industry classification: {missing_classification}")
    stock = add_stock_indicators(stock, settings)
    stock = add_stock_strength(stock, settings)
    write_parquet(stock, processed / "stock_daily_features.parquet")
    industry = aggregate_group(stock, "industry", settings)
    industry = add_group_scores(industry, settings)
    write_parquet(industry, processed / "industry_daily_features.parquet")
    basic = aggregate_group(stock, "basic_industry", settings)
    basic = add_group_scores(basic, settings)
    write_parquet(basic, processed / "basic_industry_daily_features.parquet")
    print("feature build complete (Strict EQ Mainboard Only, 2-Axis Engine v2)")


if __name__ == "__main__":
    main()
