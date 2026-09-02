from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

BASIC_HISTORY_FILE = PROCESSED / "dashboard_basic_industry_history.parquet"
INDUSTRY_HISTORY_FILE = PROCESSED / "dashboard_industry_history.parquet"
STOCK_HISTORY_FILE = PROCESSED / "dashboard_stock_history.parquet"
TOP_BUY_FILE = PROCESSED / "dashboard_top_buy_candidates.parquet"
IPO_WATCH_FILE = PROCESSED / "dashboard_ipo_watchlist.parquet"
METADATA_FILE = PROCESSED / "dashboard_metadata.json"
SYNC_FILE = PROCESSED / "last_sync.txt"
SMALL_GROUP_LIMIT = 5

st.set_page_config(
    page_title="NSE Sectoral Breadth & Buy Setups",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1550px; padding-top: 0.85rem; padding-bottom: 2rem; }
    h1 { letter-spacing: -0.035em; margin-bottom: 0.05rem; }
    h2, h3 { letter-spacing: -0.02em; }
    [data-testid="stMetric"] { padding: 0.3rem 0.45rem; }
    [data-testid="stDataFrame"] { border: 1px solid #e5e7eb; border-radius: 8px; }
    div[data-baseweb="select"] > div { min-height: 34px; }
    div[data-testid="stDateInput"] label { display: none; }
    div[data-testid="stDateInput"] { margin-top: 0.1rem; }
    div[data-testid="stButton"] > button { min-height: 34px; padding: 0.1rem 0.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unclassified"
    text = str(value).strip()
    return text if text else "Unclassified"


def as_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def is_fraction_series(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return not numeric.empty and numeric.abs().quantile(0.99) <= 1.01


def fmt_int(value: object) -> str:
    number = as_number(value)
    return "—" if number is None else f"{int(round(number)):,}"


def fmt_num(value: object, decimals: int = 2) -> str:
    number = as_number(value)
    return "—" if number is None else f"{number:,.{decimals}f}"


def fmt_pct(value: object, source_is_fraction: bool) -> str:
    number = as_number(value)
    if number is None:
        return "—"
    if source_is_fraction:
        number *= 100.0
    return f"{number:,.2f}%"


def heat_color(value: object) -> str:
    score = as_number(value)
    if score is None:
        return "background-color: #f8fafc; color: #64748b;"
    if score >= 80:
        return "background-color: #14532d; color: #ffffff; font-weight: 700;"
    if score >= 70:
        return "background-color: #166534; color: #ffffff; font-weight: 700;"
    if score >= 60:
        return "background-color: #22c55e; color: #052e16; font-weight: 700;"
    if score >= 50:
        return "background-color: #86efac; color: #14532d; font-weight: 700;"
    if score >= 40:
        return "background-color: #fef3c7; color: #78350f; font-weight: 700;"
    if score >= 30:
        return "background-color: #fed7aa; color: #7c2d12; font-weight: 700;"
    return "background-color: #fecaca; color: #7f1d1d; font-weight: 700;"


def actionability_color(value: object) -> str:
    pct = as_number(value)
    if pct is None:
        return ""
    if pct >= 20:
        return "background-color: #1e3a8a; color: #ffffff; font-weight: 700;"
    if pct > 0:
        return "background-color: #bfdbfe; color: #1e3a8a; font-weight: 700;"
    return "color: #94a3b8;"


def style_with_heatmap(raw: pd.DataFrame, display: pd.DataFrame):
    styles = pd.DataFrame("", index=display.index, columns=display.columns)
    if "Leadership Score" in raw.columns and "Leadership Score" in display.columns:
        colors = raw["Leadership Score"].map(heat_color)
        styles.loc[:, "Leadership Score"] = colors
    if "Strength" in raw.columns and "Strength" in display.columns:
        colors = raw["Strength"].map(heat_color)
        styles.loc[:, "Strength"] = colors
        if "Regime" in display.columns:
            styles.loc[:, "Regime"] = colors
    if "Actionability (Setup %)" in raw.columns and "Actionability (Setup %)" in display.columns:
        styles.loc[:, "Actionability (Setup %)"] = raw["Actionability (Setup %)"].map(actionability_color)
    if "Buy Setup Score" in raw.columns and "Buy Setup Score" in display.columns:
        styles.loc[:, "Buy Setup Score"] = raw["Buy Setup Score"].map(heat_color)
    return display.style.apply(lambda _: styles, axis=None)


def style_metric_heatmap(raw: pd.DataFrame, display: pd.DataFrame, column: str):
    styles = pd.DataFrame("", index=display.index, columns=display.columns)
    if column in raw.columns and column in display.columns:
        numeric = pd.to_numeric(raw[column], errors="coerce")
        valid = numeric.dropna()
        if not valid.empty:
            low = valid.min()
            high = valid.max()
            spread = max(high - low, 1e-9)

            def color(value: object) -> str:
                number = as_number(value)
                if number is None:
                    return ""
                ratio = (number - low) / spread
                if ratio >= 0.8:
                    return "background-color: #14532d; color: #ffffff; font-weight: 700;"
                if ratio >= 0.6:
                    return "background-color: #22c55e; color: #052e16; font-weight: 700;"
                if ratio >= 0.4:
                    return "background-color: #bbf7d0; color: #14532d; font-weight: 700;"
                if ratio >= 0.2:
                    return "background-color: #fef3c7; color: #78350f;"
                return "background-color: #f8fafc; color: #475569;"

            styles.loc[:, column] = numeric.map(color)
    return display.style.apply(lambda _: styles, axis=None)


@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def ensure_group_columns(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    data = frame.copy()
    if group_column not in data.columns:
        data[group_column] = "Unclassified"
    if "regime" not in data.columns:
        data["regime"] = "Unclassified"
    data[group_column] = data[group_column].map(clean_text)
    data["regime"] = data["regime"].map(clean_text)
    if "leadership_score" not in data.columns and "strength_score" in data.columns:
        data["leadership_score"] = data["strength_score"]
    if "actionability_score" not in data.columns:
        data["actionability_score"] = 0.0
    return data


def ensure_stock_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in ["symbol", "industry", "basic_industry", "sector"]:
        if column not in data.columns:
            data[column] = "Unclassified"
        data[column] = data[column].map(clean_text)
    return data


def trading_dates(frame: pd.DataFrame) -> list[pd.Timestamp]:
    dates = pd.to_datetime(frame["date"].dropna().unique())
    return sorted(pd.Timestamp(date) for date in dates)


def section_header(title: str, dates: list[pd.Timestamp], key: str) -> pd.Timestamp:
    latest = pd.Timestamp(dates[-1])
    state_key = f"{key}_selected_date"
    if state_key not in st.session_state:
        st.session_state[state_key] = latest
    selected = pd.Timestamp(st.session_state[state_key])
    if selected not in dates:
        selected = latest
        st.session_state[state_key] = latest
    index = dates.index(selected)

    heading, controls = st.columns([5.7, 2.3])
    with heading:
        st.subheader(title)
    with controls:
        previous, calendar, next_button = st.columns([0.4, 1.55, 0.4])
        with previous:
            if st.button("‹", key=f"{key}_previous", disabled=index == 0, use_container_width=True):
                st.session_state[state_key] = dates[index - 1]
                st.rerun()
        with calendar:
            chosen = st.date_input(
                "Historical date",
                value=selected.date(),
                min_value=pd.Timestamp(dates[0]).date(),
                max_value=latest.date(),
                key=f"{key}_calendar",
                label_visibility="collapsed",
            )
        with next_button:
            if st.button("›", key=f"{key}_next", disabled=index == len(dates) - 1, use_container_width=True):
                st.session_state[state_key] = dates[index + 1]
                st.rerun()

    requested = pd.Timestamp(chosen)
    valid_dates = [date for date in dates if date <= requested]
    resolved = valid_dates[-1] if valid_dates else dates[0]
    if resolved != selected:
        st.session_state[state_key] = resolved
        st.rerun()
    st.caption(f"As of {resolved.strftime('%d %b %Y')}")
    return resolved


def make_group_table(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    wanted = [
        group_column, "leadership_score", "actionability_score", "regime",
        "members", "nh_nl_net", "eq_ret_1d", "eq_ret_5d", "eq_ret_20d",
        "eq_ret_60d", "pct_above_50", "pct_above_200", "acc_minus_dist",
        "breakout_count", "vcp_ready_count",
    ]
    data = frame[[column for column in wanted if column in frame.columns]].copy()
    data = data.rename(columns={
        group_column: "Basic Industry" if group_column == "basic_industry" else "Industry",
        "leadership_score": "Leadership Score",
        "actionability_score": "Actionability (Setup %)",
        "regime": "Trading State",
        "members": "Constituent Stocks",
        "nh_nl_net": "Net New Highs (%)",
        "eq_ret_1d": "1D Return",
        "eq_ret_5d": "5D Return",
        "eq_ret_20d": "20D Return",
        "eq_ret_60d": "60D Return",
        "pct_above_50": "Stocks Above 50 EMA",
        "pct_above_200": "Stocks Above 200 EMA",
        "acc_minus_dist": "Accumulation − Distribution",
        "breakout_count": "Breakouts",
        "vcp_ready_count": "VCP Ready",
    })
    if "Leadership Score" in data.columns:
        data = data.sort_values("Leadership Score", ascending=False)
    data.insert(0, "Rank", range(1, len(data) + 1))
    return data.reset_index(drop=True)


def format_group_table(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in [
        "1D Return", "5D Return", "20D Return", "60D Return",
        "Stocks Above 50 EMA", "Stocks Above 200 EMA",
    ]:
        if column in data.columns:
            data[column] = data[column].apply(lambda value: fmt_pct(value, True))
    for column in ["Rank", "Constituent Stocks", "Accumulation − Distribution", "Breakouts", "VCP Ready"]:
        if column in data.columns:
            data[column] = data[column].apply(fmt_int)
    for column in ["Leadership Score", "Actionability (Setup %)", "Net New Highs (%)"]:
        if column in data.columns:
            data[column] = data[column].apply(fmt_num)
    return data


def format_stock_table(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in [
        "1D Return", "5D Return", "20D Return", "60D Return",
        "Distance from 52W High", "6M Gain", "Candle Range", "3-Day Squeeze",
    ]:
        if column in data.columns:
            data[column] = data[column].apply(lambda value: fmt_pct(value, True))
    for column in ["Rank", "Heavy Volume Days (6M)"]:
        if column in data.columns:
            data[column] = data[column].apply(fmt_int)
    for column in [
        "Strength", "Close", "Buy Setup Score", "Current Vol vs 50D Avg",
        "50D Up/Down Vol", "14D ATR", "Avg Turnover (Cr)",
    ]:
        if column in data.columns:
            data[column] = data[column].apply(fmt_num)
    return data


def top_buy_setups_view(basic_history: pd.DataFrame, stock_history: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    st.subheader("🎯 Strong Sector Buy Setups")
    st.caption(
        "Macro leaders (Leadership ≥ 70) screened for the 5-rule gauntlet: "
        "Price > 50 > 200 EMA, ≥30% 6-month advance, resting within -1% to +5% of 10/20/50 EMA, "
        "3-day ATR squeeze (≤1.2x ATR-14), and volume dry-up (≤0.5x 50D avg)."
    )

    basic_latest = basic_history[basic_history["date"] == selected_date].copy()
    stock_latest = stock_history[stock_history["date"] == selected_date].copy()

    # Get Leading Basic Industries (Leadership >= 70)
    eligible_df = basic_latest[
        (basic_latest["members"] >= SMALL_GROUP_LIMIT)
        & (basic_latest["leadership_score"] >= 70.0)
    ].sort_values("leadership_score", ascending=False)
    top_leaders = eligible_df["basic_industry"].tolist()

    # Fallback to top 5 if entire market is below 70
    if not top_leaders:
        eligible_df = basic_latest[basic_latest["members"] >= SMALL_GROUP_LIMIT].sort_values(
            "leadership_score", ascending=False
        ).head(5)
        top_leaders = eligible_df["basic_industry"].tolist()

    # Filter Established Candidates (Triple-Gate Join)
    buy_candidates = stock_latest[
        (stock_latest["basic_industry"].isin(top_leaders))
        & (stock_latest["established_buy_setup"] == 1)
    ].copy()

    # Filter IPO Candidates
    ipo_candidates = stock_latest[
        (stock_latest["basic_industry"].isin(top_leaders))
        & (stock_latest["ipo_buy_setup"] == 1)
    ].copy()

    m_col = st.columns(4)
    m_col[0].metric("Qualified Buy Setups", fmt_int(len(buy_candidates)))
    m_col[1].metric("IPO Tight Setups (>5 Cr)", fmt_int(len(ipo_candidates)))
    m_col[2].metric("Leading Basic Industries (≥70)", fmt_int(len(top_leaders)))
    m_col[3].metric("Scan Date", selected_date.strftime("%d %b %Y"))

    st.markdown("### Top Buy Setups (Established Stocks)")
    if buy_candidates.empty:
        st.info("No established stocks currently meet all 5 criteria in leading sectors today.")
    else:
        buy_candidates = buy_candidates.sort_values(
            ["gain_6m", "stock_strength_score"], ascending=[False, False]
        ).reset_index(drop=True)
        buy_candidates.insert(0, "Rank", range(1, len(buy_candidates) + 1))

        # TradingView Clickable Chart Links
        buy_candidates["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + buy_candidates["symbol"].astype(str)

        display_buy = buy_candidates.rename(columns={
            "symbol": "Symbol",
            "basic_industry": "Basic Industry",
            "close": "Close",
            "gain_6m": "6M Gain",
            "up_down_ratio": "50D Up/Down Vol",
            "atr_14": "14D ATR",
            "tight_3d_range": "3-Day Squeeze",
            "stock_strength_score": "Strength",
        })

        keep_cols = [
            "Rank", "Symbol", "Chart", "Basic Industry", "Close", "6M Gain",
            "50D Up/Down Vol", "14D ATR", "3-Day Squeeze", "Strength",
        ]
        display_buy = display_buy[[col for col in keep_cols if col in display_buy.columns]]
        st.dataframe(
            style_with_heatmap(display_buy, format_stock_table(display_buy)),
            use_container_width=True,
            hide_index=True,
            height=380,
            column_config={
                "Chart": st.column_config.LinkColumn("TradingView", display_text="Open ↗"),
            },
        )

    st.markdown("### IPO Tight Setups (New Listings)")
    st.caption("Newly listed IPO stocks in leading industries with daily range ≤ 5% and average turnover > 5 Crore.")
    if ipo_candidates.empty:
        st.info("No newly listed IPO stocks in leading industries currently meet the 5-Crore turnover and tightness criteria.")
    else:
        ipo_candidates["Avg Turnover (Cr)"] = ipo_candidates["ipo_turnover_avg"] / 10000000.0
        ipo_candidates["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + ipo_candidates["symbol"].astype(str)
        ipo_candidates = ipo_candidates.sort_values("daily_range", ascending=True).reset_index(drop=True)
        ipo_candidates.insert(0, "Rank", range(1, len(ipo_candidates) + 1))

        display_ipo = ipo_candidates.rename(columns={
            "symbol": "Symbol",
            "basic_industry": "Basic Industry",
            "close": "Close",
            "daily_range": "Candle Range",
            "ret_20d": "20D Return",
            "stock_strength_score": "Strength",
        })
        keep_ipo = [
            "Rank", "Symbol", "Chart", "Basic Industry", "Close",
            "Candle Range", "Avg Turnover (Cr)", "20D Return", "Strength",
        ]
        display_ipo = display_ipo[[col for col in keep_ipo if col in display_ipo.columns]]
        st.dataframe(
            style_with_heatmap(display_ipo, format_stock_table(display_ipo)),
            use_container_width=True,
            hide_index=True,
            height=280,
            column_config={
                "Chart": st.column_config.LinkColumn("TradingView", display_text="Open ↗"),
            },
        )

    st.markdown("### Leading Basic Industries (2-Axis Overview)")
    raw_leaders = make_group_table(eligible_df, "basic_industry")
    st.dataframe(
        style_with_heatmap(raw_leaders, format_group_table(raw_leaders)),
        use_container_width=True,
        hide_index=True,
        height=320,
    )


def basic_industry_view(basic_history: pd.DataFrame, stock_history: pd.DataFrame) -> None:
    selected_date = section_header("Basic Industry Leadership", trading_dates(basic_history), "basic")
    selected = basic_history[basic_history["date"] == selected_date].copy()

    filters = st.columns([1.45, 0.85, 0.85])
    with filters[0]:
        regimes = st.multiselect(
            "Trading State filter",
            [
                "Fresh Leader (HUNT)",
                "Extended Leader (WAIT)",
                "Speculative Coil (AVOID)",
                "Dead (AVOID)",
                "Neutral Transition",
            ],
            default=[
                "Fresh Leader (HUNT)",
                "Extended Leader (WAIT)",
                "Speculative Coil (AVOID)",
                "Dead (AVOID)",
                "Neutral Transition",
            ],
            key="basic_regimes",
        )
    with filters[1]:
        minimum = st.number_input("Minimum stocks", min_value=1, value=SMALL_GROUP_LIMIT, step=1, key="basic_minimum")
    with filters[2]:
        ranking = st.selectbox("Ranking", ["Highest Leadership", "Lowest Leadership"], key="basic_sort")

    if regimes:
        selected = selected[selected["regime"].isin(regimes)]
    if "members" in selected.columns:
        selected = selected[selected["members"] >= minimum]
    raw_table = make_group_table(selected, "basic_industry")
    if ranking == "Lowest Leadership" and "Leadership Score" in raw_table.columns:
        raw_table = raw_table.sort_values("Leadership Score").reset_index(drop=True)
        raw_table["Rank"] = range(1, len(raw_table) + 1)
    if raw_table.empty:
        st.warning("No Basic Industries match the selected filters.")
        return

    st.dataframe(
        style_with_heatmap(raw_table, format_group_table(raw_table)),
        use_container_width=True,
        hide_index=True,
        height=410,
    )

    st.markdown("### Selected Basic Industry Details")
    selected_group = st.selectbox("Basic Industry", raw_table["Basic Industry"].tolist(), key="basic_group_selector")

    stocks = stock_history[
        (stock_history["date"] == selected_date)
        & (stock_history["basic_industry"] == selected_group)
    ].copy()
    if not stocks.empty:
        stocks["Chart"] = "https://in.tradingview.com/chart/?symbol=NSE:" + stocks["symbol"].astype(str)
        stocks = stocks.sort_values("ret_20d", ascending=False).reset_index(drop=True)
        stocks.insert(0, "Rank", range(1, len(stocks) + 1))
        disp_stocks = stocks.rename(columns={
            "symbol": "Symbol",
            "close": "Close",
            "ret_20d": "20D Return",
            "ret_60d": "60D Return",
            "gain_6m": "6M Gain",
            "stock_strength_score": "Strength",
        })
        keep_cols = ["Rank", "Symbol", "Chart", "Close", "20D Return", "60D Return", "6M Gain", "Strength"]
        disp_stocks = disp_stocks[[col for col in keep_cols if col in disp_stocks.columns]]
        st.dataframe(
            style_with_heatmap(disp_stocks, format_stock_table(disp_stocks)),
            use_container_width=True,
            hide_index=True,
            height=320,
            column_config={
                "Chart": st.column_config.LinkColumn("TradingView", display_text="Open ↗"),
            },
        )


def industry_view(industry_history: pd.DataFrame, stock_history: pd.DataFrame) -> None:
    selected_date = section_header("Industry Leadership", trading_dates(industry_history), "industry")
    selected = industry_history[industry_history["date"] == selected_date].copy()
    if "members" in selected.columns:
        selected = selected[selected["members"] >= SMALL_GROUP_LIMIT]
    raw_table = make_group_table(selected, "industry")
    if raw_table.empty:
        st.warning("No Industry data is available for this date.")
        return
    st.dataframe(
        style_with_heatmap(raw_table, format_group_table(raw_table)),
        use_container_width=True,
        hide_index=True,
        height=470,
    )


def overview_view(basic_history: pd.DataFrame, industry_history: pd.DataFrame, stock_history: pd.DataFrame) -> None:
    st.subheader("Market Breadth & 2-Axis Overview")
    latest = max(basic_history["date"].max(), industry_history["date"].max())
    basic = basic_history[basic_history["date"] == latest].copy()
    industry = industry_history[industry_history["date"] == latest].copy()
    regimes = basic["regime"].value_counts()
    metrics = st.columns(5)
    metrics[0].metric("Latest data", latest.strftime("%d %b %Y"))
    metrics[1].metric("Basic Industries", fmt_int(basic["basic_industry"].nunique()))
    metrics[2].metric("Industries", fmt_int(industry["industry"].nunique()))
    metrics[3].metric("Fresh Leaders (HUNT)", f"{regimes.get('Fresh Leader (HUNT)', 0)}")
    metrics[4].metric("Extended Leaders (WAIT)", f"{regimes.get('Extended Leader (WAIT)', 0)}")

    st.markdown("### Current Basic Industry Leadership (2-Axis Matrix)")
    eligible_basic = basic[basic["members"] >= SMALL_GROUP_LIMIT] if "members" in basic.columns else basic
    raw = make_group_table(eligible_basic, "basic_industry")
    st.dataframe(
        style_with_heatmap(raw, format_group_table(raw)),
        use_container_width=True,
        hide_index=True,
        height=480,
    )


def methodology_view() -> None:
    st.subheader("Methodology (2-Axis System)")
    st.markdown(
        """
        **Axis 1: Leadership Score (Macro Institutional Trend, 0–100)**
        *   **35% Price Velocity:** Median 20-day and 60-day equal-weighted returns percentile-ranked cross-sectionally.
        *   **35% Structural Alignment:** Percentage of constituents in full EMA alignment (20 EMA > 50 EMA > 200 EMA).
        *   **30% Institutional Volume:** 50-Day Cumulative Up/Down Volume Ratio ($\sum \text{Up Volume} / \sum \text{Down Volume}$).
        *   *Smoothed with a 3-day EWM to eliminate single-day ranking noise.*

        **Axis 2: Actionability Score (Micro Setup Density %)**
        Displays the exact raw percentage of stocks in that industry passing the **5-Rule Setup Gauntlet**:
        1.  **Trend:** Price > 50 EMA > 200 EMA.
        2.  **Prior Advance:** 6-Month Advance $\ge 30\%$.
        3.  **Strike Zone:** Price resting within -1% to +5% of 10 EMA, 20 EMA, or 50 EMA.
        4.  **Coil:** 3-Day Squeeze $(\text{Highest High} - \text{Lowest Low}) \le 1.2 \times \text{ATR}_{14}$.
        5.  **Dry-Up:** Today's volume $\le 0.5\times$ the 50-day average.

        **Top Buy Setups Filter (Triple-Gate Join)**
        Stocks are only displayed on the Top Buy Setups list if their parent industry has a Leadership Score $\ge 70$ AND the individual stock passes all 5 rules.

        **IPO Watchlist Rule**
        Stocks listed for $<150$ days with 1-day candle range $\le 5\%$ and Average Rupee Turnover $> 5$ Crore (excluding listing day).
        """
    )


def main() -> None:
    required = [BASIC_HISTORY_FILE, INDUSTRY_HISTORY_FILE, STOCK_HISTORY_FILE]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        st.error("Required dashboard files are missing. Run the data workflow first.")
        st.code("\n".join(missing))
        st.stop()

    basic_history = ensure_group_columns(load_parquet(str(BASIC_HISTORY_FILE)), "basic_industry")
    industry_history = ensure_group_columns(load_parquet(str(INDUSTRY_HISTORY_FILE)), "industry")
    stock_history = ensure_stock_columns(load_parquet(str(STOCK_HISTORY_FILE)))

    st.title("NSE Sectoral Breadth & 2-Axis Setup Engine")
    sync_text = SYNC_FILE.read_text(encoding="utf-8").strip() if SYNC_FILE.exists() else "Not available"
    st.caption(f"Data as of {sync_text.replace('T', ' ').replace('Z', ' IST')}")

    tabs = st.tabs(["🎯 Top Buy Setups", "Basic Industry", "Industry", "Overview", "Methodology"])

    with tabs[0]:
        buy_setup_date = section_header("Top Buy Setups", trading_dates(stock_history), "buys")
        top_buy_setups_view(basic_history, stock_history, buy_setup_date)
    with tabs[1]:
        basic_industry_view(basic_history, stock_history)
    with tabs[2]:
        industry_view(industry_history, stock_history)
    with tabs[3]:
        overview_view(basic_history, industry_history, stock_history)
    with tabs[4]:
        methodology_view()


if __name__ == "__main__":
    main()
