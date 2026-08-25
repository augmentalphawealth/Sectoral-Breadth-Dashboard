from __future__ import annotations

import numpy as np
import pandas as pd

from utils import load_settings, p, read_parquet_safe, write_parquet

VOLUME_SHOCK_THRESHOLD = 1.5
SMALL_GROUP_LIMIT = 5


def add_stock_indicators(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", group_keys=False)

    for win in [20, 50, 150, 200]:
        df[f"sma_{win}"] = g["close"].transform(
            lambda s: s.rolling(win, min_periods=max(5, win // 4)).mean()
        )

    df["avg_vol_20"] = g["volume"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    df["avg_val_20"] = g["turnover"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    df["ret_1d"] = g["close"].pct_change(1)

    for win in settings["analysis"]["return_windows"]:
        df[f"ret_{win}d"] = g["close"].pct_change(win)

    df["above_20"] = (df["close"] > df["sma_20"]).astype(int)
    df["above_50"] = (df["close"] > df["sma_50"]).astype(int)
    df["above_200"] = (df["close"] > df["sma_200"]).astype(int)

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
                    settings["analysis"]["breakout_lookback"],
                    min_periods=20,
                ).max()
            )
        )
        & (
            df["volume"]
            > settings["analysis"]["accumulation_volume_multiplier"]
            * df["avg_vol_20"]
        )
    ).astype(int)

    df["acc_day"] = (
        (df["close"] > g["close"].shift(1))
        & (
            df["volume"]
            > settings["analysis"]["accumulation_volume_multiplier"]
            * df["avg_vol_20"]
        )
    ).astype(int)

    df["dist_day"] = (
        (df["close"] < g["close"].shift(1))
        & (
            df["volume"]
            > settings["analysis"]["distribution_volume_multiplier"]
            * df["avg_vol_20"]
        )
    ).astype(int)

    df["volume_shock_ratio"] = np.where(
        df["avg_vol_20"] > 0,
        df["volume"] / df["avg_vol_20"],
        np.nan,
    )

    df["buy_volume_shock"] = (
        (df["volume_shock_ratio"] >= VOLUME_SHOCK_THRESHOLD)
        & (df["ret_1d"] > 0)
    ).astype(int)

    df["sell_volume_shock"] = (
        (df["volume_shock_ratio"] >= VOLUME_SHOCK_THRESHOLD)
        & (df["ret_1d"] < 0)
    ).astype(int)

    df["trend_template_pass"] = (
        (df["close"] > df["sma_150"])
        & (df["close"] > df["sma_200"])
        & (df["sma_150"] > df["sma_200"])
        & (df["sma_50"] > df["sma_150"])
        & (df["dist_52w_high"] > -0.25)
    ).astype(int)

    df["range_20"] = g.apply(
        lambda x: (
            x["high"].rolling(20, min_periods=10).max()
            / x["low"].rolling(20, min_periods=10).min()
        )
        - 1
    ).reset_index(level=0, drop=True)

    df["vcp_ready"] = (
        (df["range_20"] <= settings["analysis"]["vcp_range_contraction_threshold"])
        & (
            df["volume"]
            <= settings["analysis"]["vcp_volume_dryup_threshold"]
            * df["avg_vol_20"]
        )
        & (
            df["dist_52w_high"]
            > -settings["analysis"]["pivot_proximity_threshold"]
        )
    ).astype(int)

    return df


def aggregate_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
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
        buy_volume_shock_count=("buy_volume_shock", "sum"),
        sell_volume_shock_count=("sell_volume_shock", "sum"),
        median_volume_shock=("volume_shock_ratio", "median"),
        median_dist_52w_high=("dist_52w_high", "median"),
    ).reset_index()

    agg["acc_minus_dist"] = agg["acc_days"] - agg["dist_days"]
    agg["breakout_pct"] = np.where(
        agg["members"] > 0,
        agg["breakout_count"] / agg["members"] * 100,
        np.nan,
    )
    agg["vcp_ready_pct"] = np.where(
        agg["members"] > 0,
        agg["vcp_ready_count"] / agg["members"] * 100,
        np.nan,
    )
    agg["buy_volume_shock_pct"] = np.where(
        agg["members"] > 0,
        agg["buy_volume_shock_count"] / agg["members"] * 100,
        np.nan,
    )
    agg["sell_volume_shock_pct"] = np.where(
        agg["members"] > 0,
        agg["sell_volume_shock_count"] / agg["members"] * 100,
        np.nan,
    )
    agg["small_industry"] = (agg["members"] < SMALL_GROUP_LIMIT).astype(int)

    agg["pct_above_20"] *= 100
    agg["pct_above_50"] *= 100
    agg["pct_above_200"] *= 100
    agg["trend_template_pct"] *= 100
    agg["new_high_55_pct"] *= 100
    agg["new_high_252_pct"] *= 100

    return agg


def add_group_scores(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    w = settings["scoring"]
    df = df.copy()

    df["trend_score"] = (
        (df["eq_ret_20d"].rank(pct=True) * 40)
        + (df["eq_ret_60d"].rank(pct=True) * 40)
        + ((df["pct_above_50"] / 100) * 20)
    ) / 100 * w["trend"]

    df["breadth_score"] = (
        (df["pct_above_20"] + df["pct_above_50"] + df["pct_above_200"])
        / 300
    ) * w["breadth"]

    df["rs_score"] = (
        (df["eq_ret_20d"].rank(pct=True) + df["eq_ret_60d"].rank(pct=True))
        / 2
    ) * w["rs"]

    df["volume_score"] = (
        df["acc_minus_dist"].rank(pct=True) * w["volume"]
    )

    df["breakout_score"] = (
        (
            df["breakout_count"].rank(pct=True)
            + df["vcp_ready_count"].rank(pct=True)
        )
        / 2
    ) * w["breakout"]

    df["penalty_score"] = (
        ((df["median_dist_52w_high"] < -0.20).astype(int) * 0.4)
        + ((df["pct_above_20"] > 85).astype(int) * 0.6)
    ) * w["penalty"]

    df["strength_score"] = (
        df["trend_score"]
        + df["breadth_score"]
        + df["rs_score"]
        + df["volume_score"]
        + df["breakout_score"]
        - df["penalty_score"]
    )

    conditions = [
        (df["strength_score"] >= 70) & (df["pct_above_50"] >= 60),
        (df["strength_score"] >= 55) & (df["pct_above_20"] >= 55),
        (df["pct_above_20"] < 40) & (df["eq_ret_20d"] < 0),
        (df["pct_above_20"] > 80)
        & (df["median_dist_52w_high"] > -0.05),
    ]

    labels = ["Strong", "Emerging", "Weakening", "Exhausted"]

    df["regime"] = np.select(
        conditions,
        labels,
        default="Bottoming",
    )

    return df


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

    for column in required_price_columns:
        if column not in prices.columns:
            raise ValueError(
                f"Missing required prices column: {column}"
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

    missing_classification = stock["industry"].isna().sum()
    print(
        "Price rows without industry classification: "
        f"{missing_classification}"
    )

    stock = add_stock_indicators(stock, settings)

    write_parquet(
        stock,
        processed / "stock_daily_features.parquet",
    )

    industry = aggregate_group(stock, "industry")
    industry = add_group_scores(industry, settings)

    write_parquet(
        industry,
        processed / "industry_daily_features.parquet",
    )

    basic = aggregate_group(stock, "basic_industry")
    basic = add_group_scores(basic, settings)

    write_parquet(
        basic,
        processed / "basic_industry_daily_features.parquet",
    )

    print("feature build complete")


if __name__ == "__main__":
    main()
