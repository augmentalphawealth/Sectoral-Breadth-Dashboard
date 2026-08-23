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

st.set_page_config(
    page_title="NSE Sectoral Breadth",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 0.9rem; padding-bottom: 2rem; max-width: 1500px; }
    h1 { letter-spacing: -0.03em; margin-bottom: 0.1rem; }
    h2, h3 { letter-spacing: -0.02em; }
    [data-testid="stMetric"] { padding: 0.3rem 0.45rem; }
    [data-testid="stDataFrame"] { border: 1px solid #e6eaf0; border-radius: 8px; }
    div[data-baseweb="select"] > div { min-height: 34px; }
    div[data-testid="stDateInput"] label { display: none; }
    div[data-testid="stDateInput"] { margin-top: 0.12rem; }
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


def is_fraction_series(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    return numeric.abs().quantile(0.99) <= 1.01


def fmt_int(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{int(round(float(value))):,}"


def fmt_num(value: object, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{decimals}f}"


def fmt_pct_value(value: object, source_is_fraction: bool) -> str:
    if value is None or pd.isna(value):
        return "—"
    number = float(value)
    if source_is_fraction:
        number *= 100.0
    return f"{number:,.2f}%"


def strength_label(value: object) -> str:
    if value is None or pd.isna(value):
        return "Neutral"

    try:
        score = float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return "Neutral"

    if score >= 70:
        return "Leader"
    if score >= 55:
        return "Strong"
    if score >= 40:
        return "Neutral"
    if score >= 25:
        return "Weak"
    return "Lagging"


def strength_color(value: object) -> str:
    colors = {
        "Leader": "#047857",
        "Strong": "#2563eb",
        "Neutral": "#64748b",
        "Weak": "#ea580c",
        "Lagging": "#b91c1c",
    }
    return colors.get(strength_label(value), "#64748b")

def regime_color(value: object) -> str:
    return {
        "Strong": "#047857",
        "Emerging": "#2563eb",
        "Bottoming": "#a16207",
        "Weakening": "#ea580c",
        "Exhausted": "#b91c1c",
    }.get(clean_text(value), "#64748b")


def style_table(frame: pd.DataFrame):
    styled = frame.style
    if "Strength" in frame.columns:
        styled = styled.map(
            lambda value: f"color: {strength_color(value)}; font-weight: 700;",
            subset=["Strength"],
        )
    if "Regime" in frame.columns:
        styled = styled.map(
            lambda value: f"color: {regime_color(value)}; font-weight: 700;",
            subset=["Regime"],
        )
    return styled


def style_strength_only(frame: pd.DataFrame):
    if "Strength" not in frame.columns:
        return frame.style
    return frame.style.map(
        lambda value: f"color: {strength_color(value)}; font-weight: 700;",
        subset=["Strength"],
    )


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


def top_right_time_travel(dates: list[pd.Timestamp], key: str) -> pd.Timestamp:
    latest = pd.Timestamp(dates[-1])
    state_key = f"{key}_selected_date"
    if state_key not in st.session_state:
        st.session_state[state_key] = latest
    selected = pd.Timestamp(st.session_state[state_key])
    if selected not in dates:
        selected = latest
        st.session_state[state_key] = latest
    index = dates.index(selected)

    _, control_area = st.columns([5.6, 2.4])
    with control_area:
        previous, calendar, next_button = st.columns([0.45, 1.5, 0.45])
        with previous:
            if st.button("‹", key=f"{key}_previous", disabled=index == 0, use_container_width=True):
                st.session_state[state_key] = dates[index - 1]
                st.rerun()
        with calendar:
            selected_calendar = st.date_input(
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

    requested = pd.Timestamp(selected_calendar)
    valid_dates = [date for date in dates if date <= requested]
    resolved = valid_dates[-1] if valid_dates else dates[0]
    if resolved != selected:
        st.session_state[state_key] = resolved
        st.rerun()
    return resolved


def show_section_header(title: str, dates: list[pd.Timestamp], key: str) -> pd.Timestamp:
    left, right = st.columns([5.6, 2.4])
    with left:
        st.subheader(title)
    with right:
        latest = pd.Timestamp(dates[-1])
        state_key = f"{key}_selected_date"
        if state_key not in st.session_state:
            st.session_state[state_key] = latest
        selected = pd.Timestamp(st.session_state[state_key])
        if selected not in dates:
            selected = latest
            st.session_state[state_key] = latest
        index = dates.index(selected)
        previous, calendar, next_button = st.columns([0.45, 1.5, 0.45])
        with previous:
            if st.button("‹", key=f"{key}_previous", disabled=index == 0, use_container_width=True):
                st.session_state[state_key] = dates[index - 1]
                st.rerun()
        with calendar:
            selected_calendar = st.date_input(
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
    requested = pd.Timestamp(selected_calendar)
    valid_dates = [date for date in dates if date <= requested]
    resolved = valid_dates[-1] if valid_dates else dates[0]
    if resolved != selected:
        st.session_state[state_key] = resolved
        st.rerun()
    st.caption(f"As of {resolved.strftime('%d %b %Y')}")
    return resolved


def group_table(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    wanted = [
        group_column, "regime", "members", "strength_score", "eq_ret_1d",
        "eq_ret_5d", "eq_ret_20d", "eq_ret_60d", "pct_above_20",
        "pct_above_50", "pct_above_200", "acc_minus_dist", "breakout_count",
        "vcp_ready_count", "median_dist_52w_high",
    ]
    data = frame[[column for column in wanted if column in frame.columns]].copy()
    data = data.rename(columns={
        group_column: "Basic Industry" if group_column == "basic_industry" else "Industry",
        "regime": "Regime",
        "members": "Constituent Stocks",
        "strength_score": "Strength",
        "eq_ret_1d": "1D Return",
        "eq_ret_5d": "5D Return",
        "eq_ret_20d": "20D Return",
        "eq_ret_60d": "60D Return",
        "pct_above_20": "Stocks Above 20 DMA",
        "pct_above_50": "Stocks Above 50 DMA",
        "pct_above_200": "Stocks Above 200 DMA",
        "acc_minus_dist": "Accumulation − Distribution",
        "breakout_count": "Breakouts",
        "vcp_ready_count": "VCP Ready",
        "median_dist_52w_high": "Distance from 52W High",
    })
    if "Strength" in data.columns:
        data = data.sort_values("Strength", ascending=False)
    data.insert(0, "Rank", range(1, len(data) + 1))
    return data.reset_index(drop=True)


def stock_table(frame: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "stock_rank_in_basic_industry", "symbol", "stock_strength_score",
        "stock_strength_percentile_in_basic_industry", "close", "ret_1d", "ret_5d",
        "ret_20d", "ret_60d", "above_20", "above_50", "above_200", "dist_52w_high",
        "trend_template_pass", "acc_day", "dist_day", "breakout_55", "vcp_ready",
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
        "above_20": "Above 20 DMA",
        "above_50": "Above 50 DMA",
        "above_200": "Above 200 DMA",
        "dist_52w_high": "Distance from 52W High",
        "trend_template_pass": "Trend Template",
        "acc_day": "Accumulation Day",
        "dist_day": "Distribution Day",
        "breakout_55": "55-Day Breakout",
        "vcp_ready": "VCP Ready",
    })
    if "Rank" in data.columns:
        sort_columns = [column for column in ["Rank", "Symbol"] if column in data.columns]
        data = data.sort_values(sort_columns)
    return data.reset_index(drop=True)


def display_group(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    percentage_columns = [
        "1D Return", "5D Return", "20D Return", "60D Return",
        "Stocks Above 20 DMA", "Stocks Above 50 DMA", "Stocks Above 200 DMA",
        "Distance from 52W High",
    ]
    for column in percentage_columns:
        if column in data.columns:
            fraction = is_fraction_series(data[column])
            data[column] = data[column].apply(lambda value: fmt_pct_value(value, fraction))
    for column in ["Rank", "Constituent Stocks", "Accumulation − Distribution", "Breakouts", "VCP Ready"]:
        if column in data.columns:
            data[column] = data[column].apply(fmt_int)
    if "Strength" in data.columns:
        data["Strength"] = data["Strength"].apply(lambda value: fmt_num(value, 2))
    return data


def display_stocks(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    percentage_columns = [
        "1D Return", "5D Return", "20D Return", "60D Return",
        "Above 20 DMA", "Above 50 DMA", "Above 200 DMA", "Distance from 52W High",
    ]
    for column in percentage_columns:
        if column in data.columns:
            fraction = is_fraction_series(data[column])
            data[column] = data[column].apply(lambda value: fmt_pct_value(value, fraction))
    if "Strength Percentile" in data.columns:
        fraction = is_fraction_series(data["Strength Percentile"])
        data["Strength Percentile"] = data["Strength Percentile"].apply(
            lambda value: fmt_pct_value(value, fraction)
        )
    for column in ["Rank", "Accumulation Day", "Distribution Day"]:
        if column in data.columns:
            data[column] = data[column].apply(fmt_int)
    for column in ["Strength", "Close"]:
        if column in data.columns:
            data[column] = data[column].apply(lambda value: fmt_num(value, 2))
    if "Trend Template" in data.columns:
        data["Trend Template"] = data["Trend Template"].map(
            lambda value: "Pass" if bool(value) else "Fail"
        )
    for column in ["55-Day Breakout", "VCP Ready"]:
        if column in data.columns:
            data[column] = data[column].map(lambda value: "Yes" if bool(value) else "No")
    return data


def group_history_chart(history: pd.DataFrame, group_column: str, group: str) -> None:
    st.markdown("#### Industry trend — last 12 months")
    data = history[history[group_column] == group].copy()
    if data.empty:
        st.info("No history is available for this industry.")
        return
    cutoff = data["date"].max() - pd.Timedelta(days=365)
    data = data[data["date"] >= cutoff].sort_values("date")
    metrics = {
        "Equal-weighted strength score": "strength_score",
        "Equal-weighted 20D return": "eq_ret_20d",
        "Breadth above 50 DMA": "pct_above_50",
        "Breadth above 200 DMA": "pct_above_200",
    }
    available = {label: column for label, column in metrics.items() if column in data.columns}
    if not available:
        st.info("No chart metric is available.")
        return
    metric = st.selectbox("Industry chart metric", list(available), key="industry_chart_metric")
    values = data[["date", available[metric]]].set_index("date")
    if available[metric] in {"pct_above_50", "pct_above_200"} and is_fraction_series(values.iloc[:, 0]):
        values = values * 100.0
    st.line_chart(values, height=280, use_container_width=True)


def stock_comparison_chart(history: pd.DataFrame, group: str, selected: pd.DataFrame) -> None:
    st.markdown("#### Stock comparison — last 6 months")
    symbols = selected["symbol"].drop_duplicates().tolist()
    chosen = st.multiselect(
        "Stocks to compare",
        options=symbols,
        default=symbols[: min(3, len(symbols))],
        max_selections=3,
        key="stock_compare_symbols",
    )
    if not chosen:
        st.info("Select one to three stocks to compare.")
        return
    metrics = {
        "Strength score": "stock_strength_score",
        "20D return": "ret_20d",
        "60D return": "ret_60d",
        "Close price": "close",
    }
    available = {label: column for label, column in metrics.items() if column in history.columns}
    metric = st.selectbox("Stock chart metric", list(available), key="stock_chart_metric")
    data = history[
        (history["basic_industry"] == group)
        & history["symbol"].isin(chosen)
    ][["date", "symbol", available[metric]]].copy()
    if data.empty:
        st.info("No comparison history is available.")
        return
    chart = data.pivot(index="date", columns="symbol", values=available[metric]).sort_index()
    cutoff = chart.index.max() - pd.Timedelta(days=183)
    chart = chart[chart.index >= cutoff]
    if available[metric] in {"ret_20d", "ret_60d"} and is_fraction_series(chart.stack()):
        chart = chart * 100.0
    st.line_chart(chart, height=280, use_container_width=True)


def basic_industry_view(basic_history: pd.DataFrame, stock_history: pd.DataFrame) -> None:
    dates = trading_dates(basic_history)
    selected_date = show_section_header("Basic Industry Leadership", dates, "basic")
    selected = basic_history[basic_history["date"] == selected_date].copy()

    filters = st.columns([1.5, 0.8, 0.8])
    with filters[0]:
        regimes = st.multiselect(
            "Regime filter",
            ["Strong", "Emerging", "Bottoming", "Weakening", "Exhausted"],
            default=["Strong", "Emerging", "Bottoming", "Weakening", "Exhausted"],
            key="basic_regimes",
        )
    with filters[1]:
        minimum = st.number_input("Minimum stocks", min_value=1, value=1, step=1, key="basic_minimum")
    with filters[2]:
        ranking = st.selectbox("Ranking", ["Highest strength", "Lowest strength"], key="basic_sort")

    if regimes:
        selected = selected[selected["regime"].isin(regimes)]
    if "members" in selected.columns:
        selected = selected[selected["members"] >= minimum]

    table = group_table(selected, "basic_industry")
    if ranking == "Lowest strength" and "Strength" in table.columns:
        table = table.sort_values("Strength", ascending=True).reset_index(drop=True)
        table["Rank"] = range(1, len(table) + 1)
    if table.empty:
        st.warning("No Basic Industries match the selected filters.")
        return

    st.dataframe(
        style_table(display_group(table)),
        use_container_width=True,
        hide_index=True,
        height=405,
    )
    st.download_button(
        "Download Basic Industry table",
        table.to_csv(index=False).encode("utf-8"),
        f"basic_industry_{selected_date.strftime('%Y%m%d')}.csv",
        "text/csv",
        key="download_basic_table",
    )

    st.markdown("### Constituent stock strength")
    selected_group = st.selectbox(
        "Basic Industry",
        table["Basic Industry"].tolist(),
        key="basic_group_selector",
    )

    stock_dates = set(pd.to_datetime(stock_history["date"]).dt.normalize().unique())
    if selected_date not in stock_dates:
        st.info("Stock-level data is not available for this historical date.")
        return

    stocks = stock_history[
        (stock_history["date"] == selected_date)
        & (stock_history["basic_industry"] == selected_group)
    ].copy()
    if stocks.empty:
        st.info("No constituent stocks are available for this industry on the selected date.")
        return

    stocks_raw = stock_table(stocks)
    st.dataframe(
        style_strength_only(display_stocks(stocks_raw)),
        use_container_width=True,
        hide_index=True,
        height=410,
    )
    st.download_button(
        "Download constituent stock table",
        stocks_raw.to_csv(index=False).encode("utf-8"),
        f"{selected_group.replace('/', '_')}_stocks_{selected_date.strftime('%Y%m%d')}.csv",
        "text/csv",
        key="download_stock_table",
    )

    left, right = st.columns(2)
    with left:
        group_history_chart(basic_history, "basic_industry", selected_group)
    with right:
        stock_comparison_chart(stock_history, selected_group, stocks)


def industry_view(industry_history: pd.DataFrame) -> None:
    dates = trading_dates(industry_history)
    selected_date = show_section_header("Industry Leadership", dates, "industry")
    selected = industry_history[industry_history["date"] == selected_date].copy()
    table = group_table(selected, "industry")
    if table.empty:
        st.warning("No Industry data is available for this date.")
        return
    st.dataframe(
        style_table(display_group(table)),
        use_container_width=True,
        hide_index=True,
        height=470,
    )
    st.download_button(
        "Download Industry table",
        table.to_csv(index=False).encode("utf-8"),
        f"industry_{selected_date.strftime('%Y%m%d')}.csv",
        "text/csv",
        key="download_industry_table",
    )


def overview_view(basic_history: pd.DataFrame, industry_history: pd.DataFrame) -> None:
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
    st.markdown("#### Current Basic Industry leadership")
    table = group_table(basic, "basic_industry").head(15)
    st.dataframe(style_table(display_group(table)), use_container_width=True, hide_index=True, height=440)


def methodology_view() -> None:
    st.subheader("Methodology")
    st.markdown(
        """
        **Industry strength** is based on precomputed, equal-weighted constituent participation and return measures.
        The dashboard reads prepared EOD data and does not recalculate indicators in the browser.

        **Strength percentile** is each stock's percentile rank within its selected Basic Industry.
        A reading of 100.00% is the highest-ranked stock in that industry.

        **Stocks Above 20 / 50 / 200 DMA** represent the percentage of constituent stocks above the relevant moving average.
        Returns, breadth, and distance-from-high values display with a maximum of two decimal places.
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
        overview_view(basic_history, industry_history)
    with tabs[3]:
        methodology_view()


if __name__ == "__main__":
    main()
