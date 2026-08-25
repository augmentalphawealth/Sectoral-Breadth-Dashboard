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
METADATA_FILE = PROCESSED / "dashboard_metadata.json"
SYNC_FILE = PROCESSED / "last_sync.txt"
SMALL_GROUP_LIMIT = 5

st.set_page_config(
    page_title="NSE Sectoral Breadth",
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


def style_with_heatmap(raw: pd.DataFrame, display: pd.DataFrame):
    styles = pd.DataFrame("", index=display.index, columns=display.columns)
    if "Strength" in raw.columns and "Strength" in display.columns:
        colors = raw["Strength"].map(heat_color)
        styles.loc[:, "Strength"] = colors
        if "Regime" in display.columns:
            styles.loc[:, "Regime"] = colors
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


@st.cache_data(show_spinner=False)
def load_metadata(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def ensure_group_columns(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    data = frame.copy()
    if group_column not in data.columns:
        data[group_column] = "Unclassified"
    if "regime" not in data.columns:
        data["regime"] = "Unclassified"
    data[group_column] = data[group_column].map(clean_text)
    data["regime"] = data["regime"].map(clean_text)
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
        group_column, "regime", "members", "strength_score", "pct_high_strength",
        "eq_ret_1d", "eq_ret_5d", "eq_ret_20d", "eq_ret_60d", "pct_above_50",
        "pct_above_200", "acc_minus_dist", "breakout_count", "breakout_pct",
        "vcp_ready_count", "vcp_ready_pct", "median_dist_52w_high",
    ]
    data = frame[[column for column in wanted if column in frame.columns]].copy()
    data = data.rename(columns={
        group_column: "Basic Industry" if group_column == "basic_industry" else "Industry",
        "regime": "Regime",
        "members": "Constituent Stocks",
        "strength_score": "Strength",
        "pct_high_strength": "Stocks With Strength ≥70",
        "eq_ret_1d": "1D Return",
        "eq_ret_5d": "5D Return",
        "eq_ret_20d": "20D Return",
        "eq_ret_60d": "60D Return",
        "pct_above_50": "Stocks Above 50 DMA",
        "pct_above_200": "Stocks Above 200 DMA",
        "acc_minus_dist": "Accumulation − Distribution",
        "breakout_count": "Breakouts",
        "breakout_pct": "Breakout Participation",
        "vcp_ready_count": "VCP Ready",
        "vcp_ready_pct": "VCP-Ready Participation",
        "median_dist_52w_high": "Distance from 52W High",
    })
    if "Strength" in data.columns:
        data = data.sort_values("Strength", ascending=False)
    data.insert(0, "Rank", range(1, len(data) + 1))
    return data.reset_index(drop=True)


def make_stock_table(frame: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "stock_rank_in_basic_industry", "symbol", "stock_strength_score",
        "stock_strength_percentile_in_basic_industry", "close", "ret_1d", "ret_5d",
        "ret_20d", "ret_60d", "dist_52w_high", "trend_template_pass", "acc_day",
        "dist_day", "breakout_55", "vcp_ready",
    ]
    data = frame[[column for column in wanted if column in frame.columns]].copy()
    data = data.rename(columns={
        "stock_rank_in_basic_industry": "Rank",
        "symbol": "Symbol",
        "stock_strength_score": "Strength",
        "stock_strength_percentile_in_basic_industry": "Strength Percentile",
        "close": "Close",
        "ret_1d": "1D Return",
        "ret_5d": "5D Return",
        "ret_20d": "20D Return",
        "ret_60d": "60D Return",
        "dist_52w_high": "Distance from 52W High",
        "trend_template_pass": "Trend Template",
        "acc_day": "Accumulation Day",
        "dist_day": "Distribution Day",
        "breakout_55": "55-Day Breakout",
        "vcp_ready": "VCP Ready",
    })
    if "Rank" in data.columns:
        data = data.sort_values([column for column in ["Rank", "Symbol"] if column in data.columns])
    return data.reset_index(drop=True)


def format_group_table(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in [
        "Stocks With Strength ≥70", "1D Return", "5D Return", "20D Return", "60D Return",
        "Stocks Above 50 DMA", "Stocks Above 200 DMA", "Breakout Participation",
        "VCP-Ready Participation", "Distance from 52W High",
    ]:
        if column in data.columns:
            fraction = is_fraction_series(data[column])
            data[column] = data[column].apply(lambda value: fmt_pct(value, fraction))
    for column in ["Rank", "Constituent Stocks", "Accumulation − Distribution", "Breakouts", "VCP Ready"]:
        if column in data.columns:
            data[column] = data[column].apply(fmt_int)
    if "Strength" in data.columns:
        data["Strength"] = data["Strength"].apply(fmt_num)
    return data


def format_stock_table(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in ["1D Return", "5D Return", "20D Return", "60D Return", "Distance from 52W High"]:
        if column in data.columns:
            fraction = is_fraction_series(data[column])
            data[column] = data[column].apply(lambda value: fmt_pct(value, fraction))
    if "Strength Percentile" in data.columns:
        fraction = is_fraction_series(data["Strength Percentile"])
        data["Strength Percentile"] = data["Strength Percentile"].apply(
            lambda value: fmt_pct(value, fraction)
        )
    for column in ["Rank", "Accumulation Day", "Distribution Day"]:
        if column in data.columns:
            data[column] = data[column].apply(fmt_int)
    for column in ["Strength", "Close"]:
        if column in data.columns:
            data[column] = data[column].apply(fmt_num)
    if "Trend Template" in data.columns:
        data["Trend Template"] = data["Trend Template"].map(
            lambda value: "Pass" if bool(value) else "Fail"
        )
    for column in ["55-Day Breakout", "VCP Ready"]:
        if column in data.columns:
            data[column] = data[column].map(lambda value: "Yes" if bool(value) else "No")
    return data


def setup_state(data: pd.DataFrame) -> tuple[str, str]:
    if len(data) < 20 or "strength_score" not in data.columns:
        return "Insufficient history", "At least 20 trading days are required."

    latest = data.iloc[-1]
    score = as_number(latest.get("strength_score"))
    score_10 = as_number(latest.get("strength_ma_10"))
    score_20 = as_number(latest.get("strength_ma_20"))
    score_20d_ago = as_number(data.iloc[-min(20, len(data))].get("strength_score"))
    ret_20d = as_number(latest.get("eq_ret_20d"))
    breadth_50 = as_number(latest.get("pct_above_50"))

    if score is None or score_10 is None or score_20 is None:
        return "Insufficient history", "Strength averages are not available."

    ret_positive = ret_20d is not None and ret_20d > 0
    breadth_ok = breadth_50 is not None and (breadth_50 >= 0.5 or breadth_50 >= 50)
    above_averages = score > score_10 and score > score_20
    improving = score_10 > score_20
    gained = score_20d_ago is not None and score > score_20d_ago
    extension = score - score_20

    if score >= 80 and extension >= 10:
        return "Extended", "High strength and materially above its 20-day strength average."
    if score >= 70 and above_averages and improving and ret_positive and breadth_ok:
        return "Confirmed leadership", "Strong trend, positive return confirmation and broad participation."
    if above_averages and improving and gained:
        return "Emerging leadership", "Strength is rising above both averages after a recent improvement."
    if score > score_10 and gained:
        return "Early recovery", "Strength is improving, but the 10-day/20-day trend confirmation is incomplete."
    if score < score_10 and score_10 < score_20:
        return "Weakening", "Strength is below both its 10-day and 20-day averages."
    return "Neutral transition", "No confirmed leadership or weakness pattern yet."


def group_setup_chart(history: pd.DataFrame, group: str) -> None:
    st.markdown("### Industry Setup Trend")
    st.caption(
        "Use this as a rotation and setup view: a rising score after a base is generally more useful "
        "for VCP candidate selection than an already-extended industry."
    )

    data = history[history["basic_industry"] == group].copy().sort_values("date")
    if data.empty or "strength_score" not in data.columns:
        st.info("No strength history is available for this Basic Industry.")
        return

    data["strength_score"] = pd.to_numeric(data["strength_score"], errors="coerce")
    data["strength_ma_10"] = data["strength_score"].rolling(10, min_periods=5).mean()
    data["strength_ma_20"] = data["strength_score"].rolling(20, min_periods=10).mean()

    control_left, _ = st.columns([1.1, 6.9])
    with control_left:
        period = st.selectbox("View", ["3 months", "6 months", "9 months"], index=1, key="setup_period")
    days = {"3 months": 92, "6 months": 183, "9 months": 274}[period]

    chart_data = data[data["date"] >= data["date"].max() - pd.Timedelta(days=days)].copy()
    chart_data = chart_data.set_index("date")[["strength_score", "strength_ma_10", "strength_ma_20"]]
    chart_data = chart_data.rename(columns={
        "strength_score": "Strength Score",
        "strength_ma_10": "10D Strength Average",
        "strength_ma_20": "20D Strength Average",
    })

    state, explanation = setup_state(data)
    latest = data.iloc[-1]
    metrics = st.columns(5)
    metrics[0].metric("Setup state", state)
    metrics[1].metric("Current strength", fmt_num(latest.get("strength_score")))
    metrics[2].metric("10D average", fmt_num(latest.get("strength_ma_10")))
    metrics[3].metric("20D average", fmt_num(latest.get("strength_ma_20")))
    if "eq_ret_20d" in data.columns:
        metrics[4].metric(
            "Equal-weighted 20D return",
            fmt_pct(latest.get("eq_ret_20d"), is_fraction_series(data["eq_ret_20d"])),
        )
    else:
        metrics[4].metric("Equal-weighted 20D return", "—")

    st.caption(explanation)
    st.line_chart(chart_data, height=360, use_container_width=True)

    summary = st.columns(4)
    if "pct_above_50" in data.columns:
        summary[0].metric("Breadth above 50 DMA", fmt_pct(latest.get("pct_above_50"), is_fraction_series(data["pct_above_50"])))
    if "pct_above_200" in data.columns:
        summary[1].metric("Breadth above 200 DMA", fmt_pct(latest.get("pct_above_200"), is_fraction_series(data["pct_above_200"])))
    if "pct_high_strength" in data.columns:
        summary[2].metric("Stocks with strength ≥70", fmt_pct(latest.get("pct_high_strength"), is_fraction_series(data["pct_high_strength"])))
    if "breakout_count" in data.columns:
        summary[3].metric("Current breakouts", fmt_int(latest.get("breakout_count")))


def stock_chart(history: pd.DataFrame, group: str, selected_stocks: pd.DataFrame) -> None:
    st.markdown("### Top-Stock Comparison")
    st.caption("Last 6 months. Default selection is the five highest-strength stocks in the selected Basic Industry.")
    top_symbols = (
        selected_stocks.sort_values("stock_strength_score", ascending=False)["symbol"].drop_duplicates().head(5).tolist()
        if "stock_strength_score" in selected_stocks.columns
        else selected_stocks["symbol"].drop_duplicates().head(5).tolist()
    )
    symbols = selected_stocks["symbol"].drop_duplicates().tolist()
    chosen = st.multiselect(
        "Stocks to compare",
        options=symbols,
        default=top_symbols,
        max_selections=5,
        key="stock_compare_symbols",
    )
    if not chosen:
        st.info("Select up to five stocks to compare.")
        return
    options = {
        "Strength score": "stock_strength_score",
        "20D return": "ret_20d",
        "60D return": "ret_60d",
        "Close price": "close",
    }
    available = {label: column for label, column in options.items() if column in history.columns}
    selected = st.selectbox("Stock chart metric", list(available), key="stock_chart_metric")
    data = history[
        (history["basic_industry"] == group)
        & history["symbol"].isin(chosen)
    ][["date", "symbol", available[selected]]].copy()
    if data.empty:
        st.info("No history is available for the selected stocks.")
        return
    chart = data.pivot(index="date", columns="symbol", values=available[selected]).sort_index()
    cutoff = chart.index.max() - pd.Timedelta(days=183)
    chart = chart[chart.index >= cutoff]
    if available[selected] in {"ret_20d", "ret_60d"} and is_fraction_series(chart.stack()):
        chart = chart * 100.0
    st.line_chart(chart, height=360, use_container_width=True)


def basic_industry_view(basic_history: pd.DataFrame, stock_history: pd.DataFrame) -> None:
    selected_date = section_header("Basic Industry Leadership", trading_dates(basic_history), "basic")
    selected = basic_history[basic_history["date"] == selected_date].copy()

    filters = st.columns([1.45, 0.85, 0.85])
    with filters[0]:
        regimes = st.multiselect(
            "Regime filter",
            ["Strong", "Emerging", "Bottoming", "Weakening", "Exhausted"],
            default=["Strong", "Emerging", "Bottoming", "Weakening", "Exhausted"],
            key="basic_regimes",
        )
    with filters[1]:
        minimum = st.number_input("Minimum stocks", min_value=1, value=SMALL_GROUP_LIMIT, step=1, key="basic_minimum")
    with filters[2]:
        ranking = st.selectbox("Ranking", ["Highest strength", "Lowest strength"], key="basic_sort")

    if regimes:
        selected = selected[selected["regime"].isin(regimes)]
    if "members" in selected.columns:
        selected = selected[selected["members"] >= minimum]

    raw_table = make_group_table(selected, "basic_industry")
    if ranking == "Lowest strength" and "Strength" in raw_table.columns:
        raw_table = raw_table.sort_values("Strength", ascending=True).reset_index(drop=True)
        raw_table["Rank"] = range(1, len(raw_table) + 1)
    if raw_table.empty:
        st.warning("No Basic Industries match the selected filters.")
        return

    st.dataframe(style_with_heatmap(raw_table, format_group_table(raw_table)), use_container_width=True, hide_index=True, height=410)
    st.download_button(
        "Download Basic Industry table",
        raw_table.to_csv(index=False).encode("utf-8"),
        f"basic_industry_{selected_date.strftime('%Y%m%d')}.csv",
        "text/csv",
        key="download_basic_table",
    )

    small = basic_history[basic_history["date"] == selected_date].copy()
    if "members" in small.columns:
        small = small[small["members"] < SMALL_GROUP_LIMIT]
    if not small.empty:
        with st.expander(f"Small Industries ({len(small)}) — fewer than {SMALL_GROUP_LIMIT} stocks", expanded=False):
            small_raw = make_group_table(small, "basic_industry")
            st.dataframe(style_with_heatmap(small_raw, format_group_table(small_raw)), use_container_width=True, hide_index=True, height=260)

    st.markdown("### Selected Basic Industry")
    selected_group = st.selectbox("Basic Industry", raw_table["Basic Industry"].tolist(), key="basic_group_selector")
    group_setup_chart(basic_history, selected_group)

    available_stock_dates = set(pd.to_datetime(stock_history["date"]).dt.normalize().unique())
    if selected_date not in available_stock_dates:
        st.info("Stock-level data is not available for this historical date.")
        return

    stocks = stock_history[
        (stock_history["date"] == selected_date)
        & (stock_history["basic_industry"] == selected_group)
    ].copy()
    if stocks.empty:
        st.info("No constituent stocks are available for this Basic Industry on the selected date.")
        return

    st.markdown("### Constituent Stock Strength")
    raw_stocks = make_stock_table(stocks)
    st.dataframe(style_with_heatmap(raw_stocks, format_stock_table(raw_stocks)), use_container_width=True, hide_index=True, height=410)
    st.download_button(
        "Download constituent stock table",
        raw_stocks.to_csv(index=False).encode("utf-8"),
        f"{selected_group.replace('/', '_')}_stocks_{selected_date.strftime('%Y%m%d')}.csv",
        "text/csv",
        key="download_stock_table",
    )
    stock_chart(stock_history, selected_group, stocks)


def industry_view(industry_history: pd.DataFrame) -> None:
    selected_date = section_header("Industry Leadership", trading_dates(industry_history), "industry")
    selected = industry_history[industry_history["date"] == selected_date].copy()
    raw_table = make_group_table(selected, "industry")
    if raw_table.empty:
        st.warning("No Industry data is available for this date.")
        return
    st.dataframe(style_with_heatmap(raw_table, format_group_table(raw_table)), use_container_width=True, hide_index=True, height=470)
    st.download_button(
        "Download Industry table",
        raw_table.to_csv(index=False).encode("utf-8"),
        f"industry_{selected_date.strftime('%Y%m%d')}.csv",
        "text/csv",
        key="download_industry_table",
    )


def compact_panel_table(raw: pd.DataFrame, columns: list[str], percent_columns: list[str] | None = None) -> pd.DataFrame:
    display = raw[[column for column in columns if column in raw.columns]].copy()
    percent_columns = percent_columns or []
    for column in percent_columns:
        if column in display.columns:
            fraction = is_fraction_series(display[column])
            display[column] = display[column].apply(lambda value: fmt_pct(value, fraction))
    for column in ["Constituent Stocks", "Breakouts", "VCP Ready", "Buy Shock Stocks", "Sell Shock Stocks", "70+ Stocks"]:
        if column in display.columns:
            display[column] = display[column].apply(fmt_int)
    if "Strength" in display.columns:
        display["Strength"] = display["Strength"].apply(fmt_num)
    return display


def volume_shock_panels(basic: pd.DataFrame) -> None:
    required = {"buy_volume_shock_pct", "sell_volume_shock_pct"}
    if not required.issubset(basic.columns):
        st.info("Volume shock data will appear after the next successful EOD rebuild.")
        return

    data = basic.copy()
    if "members" in data.columns:
        data = data[data["members"] >= SMALL_GROUP_LIMIT]
    data = data.rename(columns={
        "basic_industry": "Basic Industry",
        "members": "Constituent Stocks",
        "strength_score": "Strength",
        "buy_volume_shock_count": "Buy Shock Stocks",
        "sell_volume_shock_count": "Sell Shock Stocks",
        "buy_volume_shock_pct": "Buy Volume Shock",
        "sell_volume_shock_pct": "Sell Volume Shock",
        "median_volume_shock": "Median Volume Shock",
        "eq_ret_1d": "1D Return",
    })

    left, right = st.columns(2)
    with left:
        st.markdown("#### Buying Volume Shock Leaders")
        top_buy = data.sort_values("Buy Volume Shock", ascending=False).head(5)
        buy_display = compact_panel_table(
            top_buy,
            ["Basic Industry", "Buy Shock Stocks", "Buy Volume Shock", "Median Volume Shock", "1D Return", "Strength"],
            ["Buy Volume Shock", "1D Return"],
        )
        st.dataframe(style_metric_heatmap(top_buy, buy_display, "Buy Volume Shock"), use_container_width=True, hide_index=True, height=230)
    with right:
        st.markdown("#### Selling Volume Shock Leaders")
        top_sell = data.sort_values("Sell Volume Shock", ascending=False).head(5)
        sell_display = compact_panel_table(
            top_sell,
            ["Basic Industry", "Sell Shock Stocks", "Sell Volume Shock", "Median Volume Shock", "1D Return", "Strength"],
            ["Sell Volume Shock", "1D Return"],
        )
        st.dataframe(style_metric_heatmap(top_sell, sell_display, "Sell Volume Shock"), use_container_width=True, hide_index=True, height=230)


def breakout_panel(basic: pd.DataFrame) -> None:
    if "breakout_pct" not in basic.columns:
        st.info("Breakout participation data will appear after the next successful EOD rebuild.")
        return

    data = basic.copy()
    if "members" in data.columns:
        data = data[data["members"] >= SMALL_GROUP_LIMIT]
    data = data.rename(columns={
        "basic_industry": "Basic Industry",
        "members": "Constituent Stocks",
        "strength_score": "Strength",
        "breakout_count": "Breakouts",
        "breakout_pct": "Breakout Participation",
        "vcp_ready_count": "VCP Ready",
        "vcp_ready_pct": "VCP-Ready Participation",
        "eq_ret_20d": "20D Return",
    })
    top = data.sort_values(["Breakout Participation", "VCP-Ready Participation"], ascending=False).head(10)
    display = compact_panel_table(
        top,
        ["Basic Industry", "Constituent Stocks", "Breakouts", "Breakout Participation", "VCP Ready", "VCP-Ready Participation", "20D Return", "Strength"],
        ["Breakout Participation", "VCP-Ready Participation", "20D Return"],
    )
    st.dataframe(style_metric_heatmap(top, display, "Breakout Participation"), use_container_width=True, hide_index=True, height=330)


def high_strength_panel(stock_history: pd.DataFrame) -> None:
    latest_date = stock_history["date"].max()
    data = stock_history[stock_history["date"] == latest_date].copy()
    if "stock_strength_score" not in data.columns:
        st.info("High-strength stocks will appear after the next successful EOD rebuild.")
        return

    data["stock_strength_score"] = pd.to_numeric(data["stock_strength_score"], errors="coerce")
    high = data[data["stock_strength_score"] >= 70].copy()
    if high.empty:
        st.info("No stocks currently meet the Strength ≥70 threshold.")
        return

    group_summary = (
        high.groupby("basic_industry", dropna=False)
        .agg(
            **{
                "70+ Stocks": ("symbol", "nunique"),
                "Average Strength": ("stock_strength_score", "mean"),
                "Symbols": ("symbol", lambda values: ", ".join(sorted(set(values))[:12])),
            }
        )
        .reset_index()
        .rename(columns={"basic_industry": "Basic Industry"})
        .sort_values(["70+ Stocks", "Average Strength"], ascending=False)
    )
    group_summary["Average Strength"] = group_summary["Average Strength"].apply(fmt_num)
    st.dataframe(group_summary, use_container_width=True, hide_index=True, height=380)


def overview_view(basic_history: pd.DataFrame, industry_history: pd.DataFrame, stock_history: pd.DataFrame) -> None:
    st.subheader("Market Breadth Overview")
    latest = max(basic_history["date"].max(), industry_history["date"].max())
    basic = basic_history[basic_history["date"] == latest].copy()
    industry = industry_history[industry_history["date"] == latest].copy()
    regimes = basic["regime"].value_counts()

    metrics = st.columns(5)
    metrics[0].metric("Latest data", latest.strftime("%d %b %Y"))
    metrics[1].metric("Basic Industries", fmt_int(basic["basic_industry"].nunique()))
    metrics[2].metric("Industries", fmt_int(industry["industry"].nunique()))
    metrics[3].metric("Strong / Emerging", f"{regimes.get('Strong', 0)} / {regimes.get('Emerging', 0)}")
    metrics[4].metric("Weakening / Exhausted", f"{regimes.get('Weakening', 0)} / {regimes.get('Exhausted', 0)}")

    st.markdown("### Current Basic Industry Leadership")
    raw = make_group_table(basic, "basic_industry").head(15)
    st.dataframe(style_with_heatmap(raw, format_group_table(raw)), use_container_width=True, hide_index=True, height=440)

    st.markdown("### Volume Shock Leaders")
    volume_shock_panels(basic)

    st.markdown("### Breakout Participation Leaders")
    breakout_panel(basic)

    st.markdown("### Stocks With Strength ≥70")
    high_strength_panel(stock_history)


def methodology_view() -> None:
    st.subheader("Methodology")
    st.markdown(
        """
        **Industry strength** is based on precomputed, equal-weighted constituent participation and return measures.
        This dashboard reads prepared EOD data and does not recalculate indicators in the browser.

        **Industry Setup Trend** shows daily Strength Score together with its 10-day and 20-day averages.
        It is intended to identify recovery, emerging leadership, confirmation, extension, and weakening phases.
        It is a group-selection aid; individual VCP structure, pivot, volume and risk still determine trade entry.

        **Volume shock** identifies stocks trading at least 1.5 times their 20-day average volume. Buying shock requires a positive daily return; selling shock requires a negative daily return.

        **High-strength participation** measures the percentage of constituent stocks with Strength ≥70.
        This supports identifying broad industry leadership rather than leadership driven by only one or two stocks.

        **Heatmap colour** is driven by the underlying Strength score: dark green is strongest; lighter green,
        amber, orange, and red represent progressively weaker scores.

        **Stocks Above 50 DMA / 200 DMA** show industry-level participation above those moving averages.
        Small Industries with fewer than five constituents are separated to reduce ranking noise.
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

    st.title("NSE Sectoral Breadth")
    sync_text = SYNC_FILE.read_text(encoding="utf-8").strip() if SYNC_FILE.exists() else "Not available"
    st.caption(f"Data as of {sync_text.replace('T', ' ').replace('Z', ' IST')}")

    tabs = st.tabs(["Basic Industry", "Industry", "Overview", "Methodology"])
    with tabs[0]:
        basic_industry_view(basic_history, stock_history)
    with tabs[1]:
        industry_view(industry_history)
    with tabs[2]:
        overview_view(basic_history, industry_history, stock_history)
    with tabs[3]:
        methodology_view()


if __name__ == "__main__":
    main()
