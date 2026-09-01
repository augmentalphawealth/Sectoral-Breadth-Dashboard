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
    volume_shock_threshold = analysis.get(
        "volume_shock_threshold",
        DEFAULT_VOLUME_SHOCK_THRESHOLD,
    )
    g = df.groupby("symbol", group_keys=False)

    # 1. Exponential Moving Averages (EMAs)
    for span in [10, 20, 50, 150, 200]:
        df[f"ema_{span}"] = g["close"].transform(
            lambda s: s.ewm(span=span, min_periods=max(5, span // 4)).mean()
        )
        df[f"sma_{span}"] = df[f"ema_{span}"]

    # 2. Volume and Turnover Averages
    df["avg_vol_20"] = g["volume"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    df["avg_vol_50"] = g["volume"].transform(
        lambda s: s.rolling(50, min_periods=15).mean()
    )
    df["avg_val_20"] = g["turnover"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    df["ret_1d"] = g["close"].pct_change(1)

    for win in analysis.get("return_windows", [5, 10, 20, 60]):
        df[f"ret_{win}d"] = g["close"].pct_change(win)

    df["above_20"] = (df["close"] > df["ema_20"]).astype(int)
    df["above_50"] = (df["close"] > df["ema_50"]).astype(int)
    df["above_200"] = (df["close"] > df["ema_200"]).astype(int)

    # 3. 52-Week High and Low Metrics
    df["dist_52w_high"] = g["close"].transform(
        lambda s: s / s.rolling(252, min_periods=60).max() - 1
    )

    df["new_high_55"] = (
        df["close"]
        >= g["high"].transform(
            lambda s: s.rolling(55, min_periods=20).max()
        )
    ).astype(int)

    df["new_high_252"] = (
        df["close"]
        >= g["high"].transform(
            lambda s: s.rolling(252, min_periods=60).max()
        )
    ).astype(int)

    df["breakout_55"] = (
        (
            df["close"]
            > g["high"].transform(
                lambda s: s.shift(1).rolling(
                    analysis.get("breakout_lookback", 55),
                    min_periods=20,
                ).max()
            )
        )
        & (
            df["volume"]
            > analysis.get("accumulation_volume_multiplier", 1.2) * df["avg_vol_20"]
        )
    ).astype(int)

    df["acc_day"] = (
        (df["close"] > g["close"].shift(1))
        & (
            df["volume"]
            > analysis.get("accumulation_volume_multiplier", 1.2) * df["avg_vol_20"]
        )
    ).astype(int)

    df["dist_day"] = (
        (df["close"] < g["close"].shift(1))
        & (
            df["volume"]
            > analysis.get("distribution_volume_multiplier", 1.2) * df["avg_vol_20"]
        )
    ).astype(int)

    df["volume_shock_ratio"] = np.where(
        df["avg_vol_20"] > 0,
        df["volume"] / df["avg_vol_20"],
        np.nan,
    )

    df["buy_volume_shock"] = (
        (df["volume_shock_ratio"] >= volume_shock_threshold)
        & (df["ret_1d"] > 0)
    ).astype(int)

    df["sell_volume_shock"] = (
        (df["volume_shock_ratio"] >= volume_shock_threshold)
        & (df["ret_1d"] < 0)
    ).astype(int)

    # 4. Trend Template (EMA Stacking Order)
    df["trend_template_pass"] = (
        (df["close"] > df["ema_50"])
        & (df["close"] > df["ema_200"])
        & (df["ema_20"] > df["ema_50"])
        & (df["ema_50"] > df["ema_200"])
        & (df["dist_52w_high"] > -0.25)
    ).astype(int)

    # 5. Price Tightness Calculations (Multi-Day Squeeze, ADR Compression, Consecutive Closes)
    df["daily_range"] = np.where(df["close"] > 0, (df["high"] - df["low"]) / df["close"], np.nan)
    
    roll_high_5 = g["high"].transform(lambda s: s.rolling(5, min_periods=3).max())
    roll_low_5 = g["low"].transform(lambda s: s.rolling(5, min_periods=3).min())
    df["tightness_squeeze_5d"] = (roll_high_5 / roll_low_5) - 1.0
    df["tight_squeeze_pass"] = (df["tightness_squeeze_5d"] <= 0.08).astype(int)

    adr_20 = g["daily_range"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    adr_5 = g["daily_range"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    df["tight_adr_pass"] = ((adr_5 <= 0.5 * adr_20) & (df["daily_range"] <= 0.05)).astype(int)

    tight_single_day = (df["daily_range"] <= 0.05).astype(int)
    df["tight_consecutive_pass"] = (
        g["daily_range"].transform(
            lambda s: (s <= 0.05).rolling(3, min_periods=3).min()
        ) == 1
    ).astype(int)

    df["price_tightness_pass"] = (
        (tight_single_day == 1)
        | (df["tight_squeeze_pass"] == 1)
        | (df["tight_adr_pass"] == 1)
        | (df["tight_consecutive_pass"] == 1)
    ).astype(int)

    # Legacy VCP Support
    df["range_20"] = g.apply(
        lambda x: (
            x["high"].rolling(20, min_periods=10).max()
            / x["low"].rolling(20, min_periods=10).min()
        )
        - 1
    ).reset_index(level=0, drop=True)

    df["vcp_ready"] = (
        (df["range_20"] <= analysis.get("vcp_range_contraction_threshold", 0.08))
        & (
            df["volume"]
            <= analysis.get("vcp_volume_dryup_threshold", 0.65) * df["avg_vol_20"]
        )
        & (
            df["dist_52w_high"]
            > -analysis.get("pivot_proximity_threshold", 0.05)
        )
    ).astype(int)

    # 6. Prior 6-Month Move (125 Days) & Volume Dry-Up Verification
    roll_min_125 = g["low"].transform(lambda s: s.rolling(125, min_periods=25).min())
    df["gain_6m"] = np.where(roll_min_125 > 0, (df["close"] / roll_min_125) - 1.0, 0.0)

    df["vol_ratio_50"] = np.where(df["avg_vol_50"] > 0, df["volume"] / df["avg_vol_50"], np.nan)
    vol_2x_hit = (df["volume"] >= 2.0 * df["avg_vol_50"]).astype(int)
    df["vol_2x_count_6m"] = g["volume"].transform(
        lambda s: vol_2x_hit.loc[s.index].rolling(125, min_periods=25).sum()
    )
    df["max_vol_ratio_6m"] = g["vol_ratio_50"].transform(
        lambda s: s.rolling(125, min_periods=25).max()
    )

    df["vol_dryup_pass"] = (df["volume"] <= 0.5 * df["avg_vol_50"]).astype(int)

    # 7. IPO vs Established Classification
    history_count = g["close"].transform(lambda s: s.rolling(200, min_periods=1).count())
    df["is_ipo"] = (history_count < 150).astype(int)

    # Buy Setup Flags
    df["established_buy_setup"] = (
        (df["is_ipo"] == 0)
        & (df["trend_template_pass"] == 1)
        & (df["gain_6m"] >= 0.20)
        & (df["vol_2x_count_6m"] >= 2)
        & (df["vol_dryup_pass"] == 1)
        & (df["price_tightness_pass"] == 1)
    ).astype(int)

    df["ipo_buy_setup"] = (
        (df["is_ipo"] == 1)
        & (df["price_tightness_pass"] == 1)
    ).astype(int)

    return df


def add_stock_strength(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    data = df.copy()
    threshold = settings.get("analysis", {}).get(
        "high_strength_threshold",
        DEFAULT_HIGH_STRENGTH_THRESHOLD,
    )

    # Rank within each trading date
    data["stock_strength_score"] = (
        data.groupby("date")["ret_20d"].rank(pct=True) * 35
        + data.groupby("date")["ret_60d"].rank(pct=True) * 35
        + data["above_50"] * 15
        + data["above_200"] * 10
        + data["breakout_55"] * 3
        + data["vcp_ready"] * 2
    )

    data["high_strength_flag"] = (
        data["stock_strength_score"] >= threshold
    ).astype(int)

    # Buy Setup Ranking Score (Point system: Gain Size + Peak Volume + Tightness)
    gain_pts = data.groupby("date")["gain_6m"].rank(pct=True) * 35
    vol_pts = data.groupby("date")["max_vol_ratio_6m"].rank(pct=True) * 35
    price_tight_pts = (1.0 - data.groupby("date")["daily_range"].rank(pct=True)) * 15
    vol_tight_pts = (1.0 - data.groupby("date")["vol_ratio_50"].rank(pct=True)) * 15

    data["buy_setup_score"] = (
        gain_pts.fillna(0) + vol_pts.fillna(0) + price_tight_pts.fillna(0) + vol_tight_pts.fillna(0)
    ).round(2)

    return data


def aggregate_group(df: pd.DataFrame, group_col: str, settings: dict) -> pd.DataFrame:
    small_group_limit = settings.get("analysis", {}).get(
        "small_industry_limit",
        DEFAULT_SMALL_GROUP_LIMIT,
    )

    agg = df.groupby(["date", group_col], dropna=False).agg(
        members=("symbol", "nunique"),
        eq_ret_1d=("ret_1d", "mean"),
        eq_ret_5d=("ret_5d", "mean"),
        eq_ret_10d=("ret_10d", "mean"),
        eq_ret_20d=("ret_20d", "mean"),
        eq_ret_60d=("ret_60d", "mean"),
        pct_above_20=("above_20", "mean"),
        pct_above_50=("above_50", "mean"),
        pct_above_200=("above_200", "mean"),
        trend_template_pct=("trend_template_pass", "mean"),
        new_high_55_pct=("new_high_55", "mean"),
        new_high_252_pct=("new_high_252", "mean"),
        acc_days=("acc_day", "sum"),
        dist_days=("dist_day", "sum"),
        breakout_count=("breakout_55", "sum"),
        vcp_ready_count=("vcp_ready", "sum"),
        high_strength_count=("high_strength_flag", "sum"),
        buy_volume_shock_count=("buy_volume_shock", "sum"),
        sell_volume_shock_count=("sell_volume_shock", "sum"),
        median_volume_shock=("volume_shock_ratio", "median"),
        median_dist_52w_high=("dist_52w_high", "median"),
    ).reset_index()

    agg["acc_minus_dist"] = agg["acc_days"] - agg["dist_days"]

    for count_column, pct_column in [
        ("breakout_count", "breakout_pct"),
        ("vcp_ready_count", "vcp_ready_pct"),
        ("high_strength_count", "pct_high_strength"),
        ("buy_volume_shock_count", "buy_volume_shock_pct"),
        ("sell_volume_shock_count", "sell_volume_shock_pct"),
    ]:
        agg[pct_column] = np.where(
            agg["members"] > 0,
            agg[count_column] / agg["members"] * 100,
            np.nan,
        )

    agg["small_industry"] = (
        agg["members"] < small_group_limit
    ).astype(int)

    agg["pct_above_20"] *= 100
    agg["pct_above_50"] *= 100
    agg["pct_above_200"] *= 100
    agg["trend_template_pct"] *= 100
    agg["new_high_55_pct"] *= 100
    agg["new_high_252_pct"] *= 100

    return agg


def add_group_scores(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    w = settings["scoring"]
    data = df.copy()

    required_weights = [
        "trend",
        "breadth",
        "rs",
        "high_strength",
        "volume",
        "breakout",
        "penalty",
    ]
    missing_weights = [key for key in required_weights if key not in w]
    if missing_weights:
        raise ValueError(
            "Missing scoring settings: "
            f"{missing_weights}"
        )

    data["trend_score"] = (
        (data.groupby("date")["eq_ret_20d"].rank(pct=True) * 40)
        + (data.groupby("date")["eq_ret_60d"].rank(pct=True) * 40)
        + ((data["pct_above_50"] / 100) * 20)
    ) / 100 * w["trend"]

    data["breadth_score"] = (
        (data["pct_above_20"] + data["pct_above_50"] + data["pct_above_200"])
        / 300
    ) * w["breadth"]

    data["rs_score"] = (
        (data.groupby("date")["eq_ret_20d"].rank(pct=True) + data.groupby("date")["eq_ret_60d"].rank(pct=True))
        / 2
    ) * w["rs"]

    data["high_strength_score"] = (
        data["pct_high_strength"] / 100
    ) * w["high_strength"]

    data["volume_score"] = (
        0.70 * data.groupby("date")["acc_minus_dist"].rank(pct=True)
        + 0.30 * (
            data.groupby("date")["buy_volume_shock_pct"].rank(pct=True)
            - data.groupby("date")["sell_volume_shock_pct"].rank(pct=True)
        )
    ) * w["volume"]

    data["breakout_score"] = (
        0.60 * (data["breakout_pct"] / 100)
        + 0.40 * (data["vcp_ready_pct"] / 100)
    ) * w["breakout"]

    data["penalty_score"] = (
        ((data["median_dist_52w_high"] < -0.20).astype(int) * 0.4)
        + ((data["pct_above_20"] > 85).astype(int) * 0.6)
    ) * w["penalty"]

    data["strength_score"] = (
        data["trend_score"]
        + data["breadth_score"]
        + data["rs_score"]
        + data["high_strength_score"]
        + data["volume_score"]
        + data["breakout_score"]
        - data["penalty_score"]
    ).clip(lower=0, upper=100)

    conditions = [
        (data["strength_score"] >= 70)
        & (data["pct_above_50"] >= 60)
        & (data["pct_high_strength"] >= 35),
        (data["strength_score"] >= 55)
        & (data["pct_above_20"] >= 55),
        (data["pct_above_20"] < 40)
        & (data["eq_ret_20d"] < 0),
        (data["pct_above_20"] > 80)
        & (data["median_dist_52w_high"] > -0.05),
    ]

    labels = ["Strong", "Emerging", "Weakening", "Exhausted"]

    data["regime"] = np.select(
        conditions,
        labels,
        default="Bottoming",
    )

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

    required_price_columns = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    missing_price_columns = [
        column for column in required_price_columns if column not in prices.columns
    ]
    if missing_price_columns:
        raise ValueError(
            "Prices file is missing required columns: "
            f"{missing_price_columns}"
        )

    if "turnover" not in prices.columns:
        prices["turnover"] = prices["close"] * prices["volume"]

    required_master_columns = [
        "symbol",
        "isin",
        "industry",
        "basic_industry",
        "sector",
        "series",
    ]
    missing_master_columns = [
        column
        for column in required_master_columns
        if column not in master.columns
    ]
    if missing_master_columns:
        raise ValueError(
            "Classified master is missing required columns: "
            f"{missing_master_columns}"
        )

    join_columns = required_master_columns.copy()
    if "mcap" in master.columns:
        join_columns.append("mcap")

    master_for_join = (
        master[join_columns]
        .drop_duplicates(subset=["symbol"])
        .copy()
    )

    stock = prices.merge(
        master_for_join,
        on="symbol",
        how="left",
    )
    stock["date"] = pd.to_datetime(stock["date"])

    if "mcap" not in stock.columns:
        stock["mcap"] = np.nan

    # =========================================================================
    # STRICT EQ-ONLY FIREWALL (PURGES BE, BZ, SME, AND UNCLASSIFIED NOISE)
    # Aligns universe strictly to Mainboard EQ equities tradable on Zerodha Kite.
    # =========================================================================
    stock["series"] = stock["series"].fillna("").astype(str).str.strip()
    stock = stock[stock["series"] == "EQ"].copy()

    missing_classification = stock["industry"].isna().sum()
    print(
        "Price rows without industry classification: "
        f"{missing_classification}"
    )

    stock = add_stock_indicators(stock, settings)
    stock = add_stock_strength(stock, settings)

    write_parquet(
        stock,
        processed / "stock_daily_features.parquet",
    )

    industry = aggregate_group(stock, "industry", settings)
    industry = add_group_scores(industry, settings)
    write_parquet(
        industry,
        processed / "industry_daily_features.parquet",
    )

    basic = aggregate_group(stock, "basic_industry", settings)
    basic = add_group_scores(basic, settings)
    write_parquet(
        basic,
        processed / "basic_industry_daily_features.parquet",
    )

    print("feature build complete (Strict EQ Mainboard Only)")


if __name__ == "__main__":
    main()
