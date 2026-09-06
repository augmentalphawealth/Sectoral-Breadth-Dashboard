# app/dashboard.py
# Streamlit dashboard for prepared Sectoral Breadth Dashboard EOD snapshots.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Sectoral Breadth Dashboard", page_icon="📈", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SNAPSHOT_ROOT = PROCESSED / "dashboard_snapshots"
HISTORY_FILES = {
    "Basic Industry": PROCESSED / "dashboard_basic_industry_history.parquet",
    "Industry": PROCESSED / "dashboard_industry_history.parquet",
    "Sector": PROCESSED / "dashboard_sector_history.parquet",
}
GROUP_COLUMNS = {
    "Basic Industry": "basic_industry",
    "Industry": "industry",
    "Sector": "sector",
}
DISPLAY_COLUMNS = [
    "Rank",
    "Symbol",
    "Chart",
    "Close",
    "20D Return",
    "60D Return",
    "6M Gain",
    "Strength",
    "Established setup",
    "IPO setup",
]


@st.cache_data(show_spinner=False)
def read_parquet(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def read_json(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def as_date_series(data: pd.DataFrame) -> pd.DataFrame:
    if "date" in data.columns:
        data = data.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    return data


def available_dates() -> list[pd.Timestamp]:
    dates_file = PROCESSED / "dashboard_dates.parquet"
    table = read_parquet(str(dates_file))
    if not table.empty and "date" in table.columns:
        values = pd.to_datetime(table["date"], errors="coerce").dropna().dt.normalize().unique()
        result = sorted(pd.Timestamp(value) for value in values)
        if result:
            return result
    if not SNAPSHOT_ROOT.exists():
        return []
    values: list[pd.Timestamp] = []
    for folder in SNAPSHOT_ROOT.iterdir():
        if folder.is_dir():
            value = pd.to_datetime(folder.name, format="%Y-%m-%d", errors="coerce")
            if not pd.isna(value):
                values.append(pd.Timestamp(value).normalize())
    return sorted(set(values))


def snapshot_path(selected_date: pd.Timestamp, filename: str) -> Path:
    return SNAPSHOT_ROOT / selected_date.strftime("%Y-%m-%d") / filename


def snapshot(selected_date: pd.Timestamp, filename: str) -> pd.DataFrame:
    return as_date_series(read_parquet(str(snapshot_path(selected_date, filename))))


def first_column(data: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in data.columns), None)


def numeric(data: pd.DataFrame, candidates: list[str], default: float = np.nan) -> pd.Series:
    column = first_column(data, candidates)
    if column is None:
        return pd.Series(default, index=data.index, dtype="float64")
    return pd.to_numeric(data[column], errors="coerce")


def group_label(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    result = data.copy()
    if group_column not in result.columns:
        result[group_column] = "Unclassified"
    result[group_column] = result[group_column].fillna("Unclassified").astype(str).str.strip().replace("", "Unclassified")
    return result


def return_pct(data: pd.DataFrame, candidates: list[str]) -> pd.Series:
    values = numeric(data, candidates)
    if values.dropna().empty:
        return values
    # Feature files may store returns as decimals or already as percentage points.
    return values * 100 if values.abs().quantile(0.95) <= 2.5 else values


def display_number(value: Any, decimals: int = 1) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):,.{decimals}f}"


def display_pct(value: Any) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.1f}%"


def group_table(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    data = group_label(data, group_column)
    if data.empty:
        return pd.DataFrame(columns=["Rank", "Group", "Leadership", "Actionability", "Breadth", "20D Return", "60D Return", "6M Gain", "Members"])

    leadership = numeric(data, ["leadership_score", "strength_score", "strength"])
    actionability = numeric(data, ["actionability_score", "buy_setup_score", "setup_score"])
    breadth = numeric(data, ["breadth_pct", "breadth", "pct_above_20d", "pct_above_50d"])
    members = numeric(data, ["member_count", "constituent_count", "stock_count", "n_stocks"])
    result = pd.DataFrame(
        {
            "Group": data[group_column],
            "Leadership": leadership,
            "Actionability": actionability,
            "Breadth": breadth,
            "20D Return": return_pct(data, ["return_20d", "ret_20d", "return20d"]),
            "60D Return": return_pct(data, ["return_60d", "ret_60d", "return60d"]),
            "6M Gain": return_pct(data, ["gain_6m", "return_6m", "ret_6m", "six_month_gain"]),
            "Members": members,
        }
    )
    result = result.drop_duplicates("Group", keep="first")
    result = result.sort_values(["Leadership", "Actionability"], ascending=[False, False], na_position="last").reset_index(drop=True)
    result.insert(0, "Rank", np.arange(1, len(result) + 1))
    return result


def stock_table(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)
    symbols = data.get("symbol", pd.Series("", index=data.index)).fillna("").astype(str).str.strip()
    result = pd.DataFrame(
        {
            "Symbol": symbols,
            "Close": numeric(data, ["close", "Close", "last_price"]),
            "20D Return": return_pct(data, ["return_20d", "ret_20d", "return20d"]),
            "60D Return": return_pct(data, ["return_60d", "ret_60d", "return60d"]),
            "6M Gain": return_pct(data, ["gain_6m", "return_6m", "ret_6m", "six_month_gain"]),
            "Strength": numeric(data, ["stock_strength_score", "strength_score", "strength"]),
            "Established setup": numeric(data, ["established_buy_setup"]).fillna(0).astype(int),
            "IPO setup": numeric(data, ["ipo_buy_setup"]).fillna(0).astype(int),
            "_priority": numeric(data, ["buy_priority_score", "stock_strength_score", "strength_score"]),
        }
    )
    result = result[result["Symbol"] != ""]
    result = result.sort_values(["_priority", "Strength", "6M Gain"], ascending=[False, False, False], na_position="last").drop(columns="_priority").reset_index(drop=True)
    result.insert(0, "Rank", np.arange(1, len(result) + 1))
    result.insert(2, "Chart", result["Symbol"].map(lambda symbol: f"https://www.tradingview.com/chart/?symbol=NSE%3A{symbol}"))
    return result[DISPLAY_COLUMNS]


def styled_stock_table(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    if output.empty:
        return output
    output["Chart"] = output["Chart"].map(lambda url: f"<a href='{url}' target='_blank'>Open ↗</a>")
    output["Close"] = output["Close"].map(lambda x: display_number(x, 2))
    for column in ["20D Return", "60D Return", "6M Gain"]:
        output[column] = output[column].map(display_pct)
    output["Strength"] = output["Strength"].map(lambda x: display_number(x, 1))
    return output


def navigate(group_type: str, group_name: str) -> None:
    st.session_state["nav_group_type"] = group_type
    st.session_state["nav_group_name"] = group_name
    st.query_params["view"] = group_type.lower().replace(" ", "-")
    st.query_params["group"] = group_name
    st.rerun()


def navigation_target(group_type: str) -> str:
    return {
        "Basic Industry": "basic-industry-constituents",
        "Industry": "industry-constituents",
        "Sector": "sector-constituents",
    }[group_type]


def navigation_script(group_type: str) -> None:
    if st.session_state.pop("scroll_target", None) != group_type:
        return
    target = navigation_target(group_type)
    st.components.v1.html(
        f"""
        <script>
        const go = () => {{
          const target = window.parent.document.getElementById({json.dumps(target)});
          if (target) target.scrollIntoView({{behavior: 'smooth', block: 'start'}});
        }};
        setTimeout(go, 150);
        </script>
        """,
        height=0,
    )


def render_group_snapshot(group_type: str, selected_date: pd.Timestamp, file_name: str) -> None:
    group_column = GROUP_COLUMNS[group_type]
    data = snapshot(selected_date, file_name)
    table = group_table(data, group_column)
    st.subheader(f"{group_type} rankings")
    st.caption("Ranked by leadership score, then actionability score. Click a row action to open the matching stock constituents and group trend.")
    if table.empty:
        st.info(f"Prepared {group_type.lower()} snapshot is not available for {selected_date:%d %b %Y}. Run the EOD workflow and refresh this page.")
        return

    left, middle, right = st.columns([1, 1, 2])
    with left:
        minimum_members = st.number_input("Minimum members", min_value=0, value=0, step=1, key=f"min_members_{group_type}")
    with middle:
        minimum_score = st.number_input("Minimum leadership", value=0.0, step=1.0, key=f"min_score_{group_type}")
    with right:
        search = st.text_input(f"Search {group_type}", key=f"search_{group_type}")

    filtered = table.copy()
    filtered = filtered[(filtered["Members"].fillna(0) >= minimum_members) & (filtered["Leadership"].fillna(-np.inf) >= minimum_score)]
    if search.strip():
        filtered = filtered[filtered["Group"].str.contains(search.strip(), case=False, na=False)]
    st.dataframe(
        filtered.style.format(
            {
                "Leadership": "{:.1f}",
                "Actionability": "{:.1f}",
                "Breadth": "{:.1f}",
                "20D Return": "{:.1f}%",
                "60D Return": "{:.1f}%",
                "6M Gain": "{:.1f}%",
                "Members": "{:.0f}",
            },
            na_rep="—",
        ),
        use_container_width=True,
        hide_index=True,
        height=min(640, 70 + 35 * max(1, len(filtered))),
    )

    st.markdown("##### Open constituents and trend")
    choices = filtered["Group"].tolist()
    if not choices:
        st.caption("No groups match the filters.")
        return
    selected = st.selectbox(f"Choose a {group_type.lower()}", choices, key=f"open_{group_type}")
    if st.button("Open constituents + chart", key=f"navigate_{group_type}"):
        navigate(group_type, selected)


def render_constituents(group_type: str, selected_date: pd.Timestamp, stock: pd.DataFrame) -> None:
    group_column = GROUP_COLUMNS[group_type]
    anchor = navigation_target(group_type)
    st.markdown(f"<div id='{anchor}'></div>", unsafe_allow_html=True)
    navigation_script(group_type)
    st.subheader(f"{group_type} constituents")
    if stock.empty or group_column not in stock.columns:
        st.info("The selected date does not have a prepared stock snapshot. Run the EOD workflow and refresh the dashboard.")
        return

    data = group_label(stock, group_column)
    choices = sorted(data[group_column].dropna().astype(str).unique().tolist())
    if not choices:
        st.info("No classified constituents are available for this selection.")
        return

    nav_type = st.session_state.get("nav_group_type")
    nav_name = st.session_state.get("nav_group_name")
    query_type = st.query_params.get("view", "")
    query_name = st.query_params.get("group", "")
    preferred = nav_name if nav_type == group_type else (query_name if query_type == group_type.lower().replace(" ", "-") else None)
    if preferred not in choices:
        preferred = choices[0]
    index = choices.index(preferred)
    selected = st.selectbox(f"Select {group_type.lower()}", choices, index=index, key=f"constituent_{group_type}")

    if nav_type == group_type:
        st.session_state["scroll_target"] = group_type
        st.session_state.pop("nav_group_type", None)
        st.session_state.pop("nav_group_name", None)

    group_stocks = data[data[group_column] == selected].copy()
    table = stock_table(group_stocks)
    st.caption(f"{len(table):,} stock(s) in {selected} on {selected_date:%d %b %Y}.")
    st.dataframe(
        styled_stock_table(table),
        use_container_width=True,
        hide_index=True,
        column_config={"Chart": st.column_config.LinkColumn("Chart", display_text="Open ↗")},
        height=min(720, 70 + 35 * max(1, len(table))),
    )
    render_group_trend(group_type, selected, selected_date)


def render_group_trend(group_type: str, group_name: str, selected_date: pd.Timestamp) -> None:
    path = HISTORY_FILES[group_type]
    group_column = GROUP_COLUMNS[group_type]
    history = as_date_series(read_parquet(str(path)))
    if history.empty or group_column not in history.columns or "date" not in history.columns:
        st.caption("Prepared group trend history is not available yet.")
        return
    history = group_label(history, group_column)
    history = history[(history[group_column] == group_name) & (history["date"] <= selected_date)].copy()
    if history.empty:
        st.caption("No trend history is available for this group and date.")
        return
    score_column = first_column(history, ["leadership_score", "strength_score", "strength"])
    if score_column is None:
        st.caption("Trend score column is unavailable.")
        return
    history[score_column] = pd.to_numeric(history[score_column], errors="coerce")
    history = history.dropna(subset=[score_column]).sort_values("date").tail(260)
    if history.empty:
        st.caption("No usable trend-score observations are available.")
        return
    chart = px.line(history, x="date", y=score_column, title=f"{group_type} leadership trend — {group_name}")
    chart.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10), yaxis_title="Leadership score", xaxis_title=None)
    st.plotly_chart(chart, use_container_width=True)


def render_top_setups(selected_date: pd.Timestamp) -> None:
    st.subheader("Top setups")
    st.caption("Prepared EOD candidate lists, ranked before the dashboard loads.")
    buy = snapshot(selected_date, "top_buy_candidates.parquet")
    ipo = snapshot(selected_date, "ipo_watchlist.parquet")
    left, right = st.columns(2)
    with left:
        st.markdown("### Established buy setups")
        table = stock_table(buy)
        if table.empty:
            st.info("No established buy setups were prepared for this date.")
        else:
            st.dataframe(styled_stock_table(table), use_container_width=True, hide_index=True, column_config={"Chart": st.column_config.LinkColumn("Chart", display_text="Open ↗")}, height=min(620, 70 + 35 * len(table)))
    with right:
        st.markdown("### IPO watchlist")
        table = stock_table(ipo)
        if table.empty:
            st.info("No IPO buy setups were prepared for this date.")
        else:
            st.dataframe(styled_stock_table(table), use_container_width=True, hide_index=True, column_config={"Chart": st.column_config.LinkColumn("Chart", display_text="Open ↗")}, height=min(620, 70 + 35 * len(table)))


def render_methodology() -> None:
    st.subheader("Methodology")
    st.markdown("""
### Stock ranking
Each stock is evaluated from the precomputed EOD feature table. The dashboard displays the available close, 20-session return, 60-session return, six-month gain, strength score, and setup flags for the selected date.

- **Strength** is the stock-level strength / leadership field calculated by the EOD feature pipeline.
- **Momentum** is represented by 20D return, 60D return, and six-month gain; these are displayed as percentages.
- **Established buy setup** and **IPO buy setup** are rule-based flags produced upstream by the EOD calculation.
- **Top setups** are ordered by the precomputed `buy_priority_score`; where supplied, that score combines range contraction, volume expansion, six-month performance, up/down-volume behaviour, and stock strength.

### Basic Industry, Industry and Sector ranking
A group’s rank is calculated in the EOD group-feature pipeline and is not recalculated in Streamlit. The dashboard sorts groups by **Leadership score** first and **Actionability score** second.

- **Leadership score** summarizes the relative market leadership of the group from its constituent stocks’ prepared breadth, momentum, and strength inputs.
- **Actionability score** emphasizes how many constituents currently meet the prepared buy-setup conditions and related tradability filters.
- **Breadth** reports the prepared share / breadth metric available for that group on the selected date.
- **Members** is the number of classified stock constituents used for that group-date observation.
- The same stock may appear under one Basic Industry, one Industry, and one Sector according to the classification master used in the EOD pipeline.

### Important interpretation notes
The dashboard is a market-breadth and screening tool, not investment advice. Scores compare groups within the available NSE classification universe and can change as prices, volumes, classifications, and listed-stock histories update. Always review the individual chart, liquidity, corporate actions, results, and risk before acting.
""")


def main() -> None:
    st.title("Sectoral Breadth Dashboard")
    dates = available_dates()
    if not dates:
        st.error("No prepared dashboard snapshots were found. Run the EOD workflow to publish data/processed/dashboard_snapshots.")
        st.stop()

    latest = dates[-1]
    metadata = read_json(str(snapshot_path(latest, "metadata.json")))
    with st.sidebar:
        st.header("Dashboard controls")
        selected_date = st.selectbox(
            "As-of date",
            dates,
            index=len(dates) - 1,
            format_func=lambda value: value.strftime("%d %b %Y"),
        )
        st.caption(f"Latest prepared date: {latest:%d %b %Y}")
        if metadata:
            st.caption(f"Latest prepared stock universe: {metadata.get('stock_rows', '—'):,}" if isinstance(metadata.get("stock_rows"), int) else "Prepared EOD snapshot")

    stock = snapshot(selected_date, "stock_snapshot.parquet")
    basic_tab, setups_tab, industry_tab, sector_tab, methodology_tab = st.tabs(
        ["Basic Industry", "Top Setups", "Industry", "Sector", "Methodology"]
    )

    with basic_tab:
        render_group_snapshot("Basic Industry", selected_date, "basic_industry_snapshot.parquet")
        st.divider()
        render_constituents("Basic Industry", selected_date, stock)

    with setups_tab:
        render_top_setups(selected_date)

    with industry_tab:
        render_group_snapshot("Industry", selected_date, "industry_snapshot.parquet")
        st.divider()
        render_constituents("Industry", selected_date, stock)

    with sector_tab:
        render_group_snapshot("Sector", selected_date, "sector_snapshot.parquet")
        st.divider()
        render_constituents("Sector", selected_date, stock)

    with methodology_tab:
        render_methodology()


if __name__ == "__main__":
    main()
