import pathlib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

@st.cache_data
def load_raw_data(path: pathlib.Path) -> pd.DataFrame:
    df_raw = pd.read_csv(path)
    return df_raw


@st.cache_data
def prepare_data_and_gaps(path: pathlib.Path):
    """Load CSV, prepare 1m data, and compute gap table once, then reuse."""
    df_raw = load_raw_data(path)
    df_1m = prepare_1m_data(df_raw)
    gap_table = compute_rth_gap_table(df_1m)
    return df_1m, gap_table

# ================== Streamlit page config ==================

st.set_page_config(page_title="RTH Gap Finder", layout="wide")

st.title("RTH Gap Finder")

st.markdown(
    """
This app uses a **built-in historical 1-minute dataset** stored with the code,
so you don't need to upload anything.

In this step, we:
1. Load the CSV from the `data` folder.
2. Prepare the 1-minute data (parse `Datetime`, set timezone).
3. Extract RTH sessions (09:30–16:14 New York time).
4. Compute daily **gaps** between today's RTH open and previous day's RTH close.
"""
)

# ================== Core helper functions ==================


def prepare_1m_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare 1-minute OHLC data:

    - parse Datetime (offset-aware strings)
    - convert to America/New_York timezone
    - set as index, sort, drop duplicates
    """
    if "Datetime" not in df_raw.columns:
        raise ValueError("CSV must contain a 'Datetime' column.")

    required_cols = ["Open", "High", "Low", "Close"]
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"CSV is missing required OHLC columns: {missing}")

    df = df_raw.copy()

    # Your sample looks like: 2007-04-01 18:01:00-04:00
    # Parse as UTC first, then convert to New York.
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")

    # Remove rows that failed parsing
    df = df.dropna(subset=["Datetime"])

    # Convert timezone to America/New_York
    df["Datetime"] = df["Datetime"].dt.tz_convert("America/New_York")

    # Use Datetime as index, sort, drop duplicate timestamps
    df = df.set_index("Datetime")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    return df


def compute_rth_gap_table(
    df_1m: pd.DataFrame,
    rth_start: str = "09:30:00",
    rth_end: str = "16:14:59",
    max_days_for_gap: int = 10,
) -> pd.DataFrame:
    """
    Build a daily RTH table and compute gaps between today's RTH open
    and previous day's RTH close.

    Returns a DataFrame indexed by session date (naive Timestamp) with:
    - rth_open, rth_close
    - prev_rth_close
    - gap_points, gap_abs_points
    - gap_pct, gap_abs_pct
    - direction ("up"/"down")
    """
    # Restrict to RTH window in New York time
    rth_df = df_1m.between_time(rth_start, rth_end)
    if rth_df.empty:
        return pd.DataFrame()

    # Aggregate per calendar day (New York date)
    daily_summary = rth_df.groupby(rth_df.index.date).agg(
        rth_open=("Open", "first"),
        rth_close=("Close", "last"),
    )

    # Index as naive Timestamp (date only, no tz)
    daily_summary.index = pd.to_datetime(daily_summary.index)
    daily_summary = daily_summary.sort_index()

    # Filter out long breaks (weekends, long holidays)
    daily_summary["days_since_prev_close"] = (
        daily_summary.index.to_series().diff().dt.days
    )

    valid = daily_summary[daily_summary["days_since_prev_close"] <= max_days_for_gap].copy()

    # Previous RTH close
    valid["prev_rth_close"] = valid["rth_close"].shift(1)
    valid = valid.dropna(subset=["prev_rth_close"])

    # Gap metrics
    valid["gap_points"] = valid["rth_open"] - valid["prev_rth_close"]
    valid["gap_abs_points"] = valid["gap_points"].abs()
    valid["gap_pct"] = valid["gap_points"] / valid["prev_rth_close"]
    valid["gap_abs_pct"] = valid["gap_pct"].abs()
    valid["direction"] = np.where(valid["gap_points"] > 0, "up", "down")

    # We don't need this column anymore
    valid = valid.drop(columns=["days_since_prev_close"])

    return valid

def build_gap_candle_figure(
    df_1m: pd.DataFrame,
    gap_table: pd.DataFrame,
    session_date: pd.Timestamp,
    include_overnight: bool = False,
) -> go.Figure:
    """
    Candlestick chart showing:
    - previous RTH session (09:30–16:14)
    - current RTH session (09:30–16:14)

    Modes:
    - RTH only: previous RTH + current RTH back-to-back.
    - RTH + overnight (ETH-style): previous RTH 09:30 → current RTH 16:14
      with ALL overnight bars in between (no visual gap).

    Uses a CATEGORY x-axis so bars are back-to-back.
    Also overlays the FIRST 3 fair value gaps (3-bar FVG) after 09:30 of the
    CURRENT RTH day, each extended for 10 minutes (10 bars).
    """

    # Ensure the date exists in gap_table
    if session_date not in gap_table.index:
        raise ValueError("Selected session date not found in gap table.")

    # Locate previous session date
    gap_idx = gap_table.index.get_loc(session_date)
    if isinstance(gap_idx, slice) or isinstance(gap_idx, np.ndarray):
        if isinstance(gap_idx, np.ndarray):
            gap_idx = int(gap_idx[0])
        elif isinstance(gap_idx, slice):
            gap_idx = int(gap_idx.start)

    if gap_idx == 0:
        prev_date = session_date
    else:
        prev_date = gap_table.index[gap_idx - 1]

    tz = df_1m.index.tz

    def rth_slice_for_date(d: pd.Timestamp) -> pd.DataFrame:
        """Return only RTH bars (09:30–16:14) for a given naive date."""
        day = d.date()
        start_dt = pd.Timestamp(day).tz_localize(tz) + pd.Timedelta(hours=9, minutes=30)
        end_dt = pd.Timestamp(day).tz_localize(tz) + pd.Timedelta(hours=16, minutes=15)  # 16:14 inclusive
        return df_1m.loc[(df_1m.index >= start_dt) & (df_1m.index < end_dt)]

    prev_slice = rth_slice_for_date(prev_date)
    curr_slice = rth_slice_for_date(session_date)

    if prev_slice.empty or curr_slice.empty:
        raise ValueError("Missing RTH data for previous or current session.")

    # --- Build combined window depending on mode ---

    if include_overnight:
        # ETH-style: take everything from prev RTH start to current RTH end
        prev_start = prev_slice.index[0]
        curr_end = curr_slice.index[-1]
        combined = df_1m.loc[(df_1m.index >= prev_start) & (df_1m.index <= curr_end)]
    else:
        # RTH only: just previous RTH + current RTH
        combined = pd.concat([prev_slice, curr_slice])

    if combined.empty:
        raise ValueError("No data available for the selected sessions.")

    # CATEGORY axis: use stringified timestamps as x values
    x_vals = combined.index.astype(str)

    # ----------------- X-axis tick labels -----------------
    # RTH-only  -> every 15 minutes
    # ETH-style -> every 1 hour
    tickvals: list[str] = []
    ticktext: list[str] = []

    for ts, s in zip(combined.index, x_vals):
        if include_overnight:
            # ETH style: label every full hour
            if ts.minute == 0 and ts.second == 0:
                tickvals.append(s)
                ticktext.append(ts.strftime("%H:%M"))
        else:
            # RTH-only: label every 15 minutes
            if ts.minute % 15 == 0 and ts.second == 0:
                tickvals.append(s)
                ticktext.append(ts.strftime("%H:%M"))

    # Fallback in weird cases: ensure at least first/last label
    if not tickvals:
        tickvals = [x_vals[0], x_vals[-1]]
        ticktext = [
            combined.index[0].strftime("%H:%M"),
            combined.index[-1].strftime("%H:%M"),
        ]

    # Gap info for quadrants / title
    gap_info = gap_table.loc[session_date]
    prev_close = float(gap_info["prev_rth_close"])
    rth_open = float(gap_info["rth_open"])
    gap_points = float(gap_info["gap_points"])
    gap_pct = float(gap_info["gap_pct"]) * 100.0  # convert to %

    # Quadrant levels between prev_close and rth_open
    low = min(prev_close, rth_open)
    high = max(prev_close, rth_open)
    span = high - low if high != low else 0.0

    level_25 = low + 0.25 * span if span != 0 else low
    level_50 = low + 0.50 * span if span != 0 else low
    level_75 = low + 0.75 * span if span != 0 else low

    # Candlestick with notebook-like colors (no whisker caps)
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=x_vals,
                open=combined["Open"],
                high=combined["High"],
                low=combined["Low"],
                close=combined["Close"],
                increasing_line_color="#9c9c9c",
                increasing_fillcolor="#9c9c9c",
                decreasing_line_color="#4a4a4a",
                decreasing_fillcolor="#4a4a4a",
                whiskerwidth=0,  # no horizontal caps on wicks
            )
        ]
    )

    fig.update_layout(
        title=(
            f"Gap Day: {session_date.date().isoformat()} "
            f" | Gap: {gap_points:.2f} pts / {gap_pct:.2f}%"
        ),
        xaxis_title=(
            "Bars (prev RTH -> current RTH)"
            if not include_overnight
            else "Bars (prev RTH -> overnight -> current RTH)"
        ),
        yaxis_title="Price",
        xaxis=dict(
            type="category",
            showgrid=False,
            tickangle=0,
            tickvals=tickvals,
            ticktext=ticktext,
        ),
        yaxis=dict(
            showgrid=False,
            side="right",
        ),
        plot_bgcolor="#0f0f0f",
        paper_bgcolor="#0f0f0f",
        font=dict(color="white"),
        height=700,
        xaxis_rangeslider_visible=False,
    )

    # ----------------- FVG DETECTION (current RTH only) -----------------

    curr = curr_slice.copy()
    fvgs = []
    extend_bars = 10  # 10 minutes on 1-min data
    max_fvgs = 3      # first 3 only

    hi = curr["High"].values
    lo = curr["Low"].values
    idx = curr.index.to_list()

    for i in range(0, len(curr) - 2):
        # 3-bar windows: bar1=i, bar2=i+1, bar3=i+2
        h1, l1 = hi[i], lo[i]
        h3, l3 = hi[i + 2], lo[i + 2]

        # Bullish FVG: low of bar3 > high of bar1
        if l3 > h1:
            start_ts = idx[i]
            end_pos = min(i + extend_bars, len(curr) - 1)
            end_ts = idx[end_pos]
            y0 = h1
            y1 = l3
            fvgs.append(
                dict(
                    kind="bull",
                    x0=str(start_ts),
                    x1=str(end_ts),
                    y0=y0,
                    y1=y1,
                )
            )

        # Bearish FVG: high of bar3 < low of bar1
        if h3 < l1:
            start_ts = idx[i]
            end_pos = min(i + extend_bars, len(curr) - 1)
            end_ts = idx[end_pos]
            y0 = h3
            y1 = l1
            fvgs.append(
                dict(
                    kind="bear",
                    x0=str(start_ts),
                    x1=str(end_ts),
                    y0=y0,
                    y1=y1,
                )
            )

        if len(fvgs) >= max_fvgs:
            break

    # Draw the FVG rectangles under candles
    for fvg in fvgs:
        if fvg["kind"] == "bull":
            fill = "rgba(0, 200, 0, 0.15)"
            edge = "rgba(0, 200, 0, 0.5)"
        else:
            fill = "rgba(200, 0, 0, 0.15)"
            edge = "rgba(200, 0, 0, 0.5)"

        fig.add_shape(
            type="rect",
            x0=fvg["x0"],
            x1=fvg["x1"],
            y0=fvg["y0"],
            y1=fvg["y1"],
            xref="x",
            yref="y",
            fillcolor=fill,
            line=dict(color=edge, width=0.5),
            layer="below",
        )

    # ----------------- Horizontal lines: prev close, open, quadrants -----------------

    fig.add_hline(
        y=prev_close,
        line=dict(color="red", width=0.8),
    )
    fig.add_hline(
        y=rth_open,
        line=dict(color="blue", width=0.8),
    )

    fig.add_hline(
        y=level_25,
        line=dict(color="white", width=0.6, dash="dot"),
        opacity=0.75,
    )
    fig.add_hline(
        y=level_50,
        line=dict(color="orange", width=1.0, dash="dot"),
        opacity=1.0,
    )
    fig.add_hline(
        y=level_75,
        line=dict(color="white", width=0.6, dash="dot"),
        opacity=0.75,
    )

    # ----------------- Vertical RTH-start marker (ETH mode only) -----------------

    if include_overnight:
        # current-day RTH starts at the first bar of curr_slice (09:30)
        rth_start_curr = curr_slice.index[0]
        rth_start_x = str(rth_start_curr)

        fig.add_vline(
            x=rth_start_x,
            line_color="white",
            line_width=1,
            opacity=0.12,  # very low opacity
        )

    return fig





# ================== Load built-in dataset ==================

# ================== Load + preprocess (cached) ==================

DATA_PATH = pathlib.Path(__file__).parent / "data" / "historical_1m.csv.gz"


st.subheader("1–3. Load data, prepare 1m, and compute RTH gaps")

if not DATA_PATH.exists():
    st.error(f"Data file not found: {DATA_PATH}")
    st.stop()

try:
    df_1m, gap_table = prepare_data_and_gaps(DATA_PATH)
except Exception as e:
    st.error(f"Error during data preparation: {e}")
    st.stop()

# Just show some info; heavy work is already cached
st.success(f"Loaded and prepared data from: {DATA_PATH.name}")
st.write("Prepared 1-minute dataset shape:", df_1m.shape)
st.write("Gap table sessions count:", len(gap_table))

with st.expander("Show first 10 prepared 1m rows"):
    st.dataframe(df_1m.head(10))

with st.expander("Show first 10 gap sessions"):
    display_cols = [
        "prev_rth_close",
        "rth_open",
        "rth_close",
        "gap_points",
        "gap_abs_points",
        "gap_pct",
        "gap_abs_pct",
        "direction",
    ]
    st.dataframe(gap_table[display_cols].head(10).round(4))


if gap_table.empty:
    st.warning("No valid RTH sessions found in the prepared data.")
    st.stop()

st.write("Total RTH sessions with a valid previous close:", len(gap_table))

display_cols = [
    "prev_rth_close",
    "rth_open",
    "rth_close",
    "gap_points",
    "gap_abs_points",
    "gap_pct",
    "gap_abs_pct",
    "direction",
]

with st.expander("Show first 15 unfiltered gap sessions"):
    st.dataframe(gap_table[display_cols].head(15).round(4))


# ================== Date & gap filters (Apply button + one-example view) ==================

st.subheader("4. Filter sessions by date and gap size")

# --- Initialize session state ---

if "filtered_df" not in st.session_state:
    st.session_state["filtered_df"] = None
if "current_idx" not in st.session_state:
    st.session_state["current_idx"] = 0
if "gap_mode" not in st.session_state:
    st.session_state["gap_mode"] = "Percent of previous RTH close"
if "date_range" not in st.session_state:
    st.session_state["date_range"] = (
        gap_table.index.min().date(),
        gap_table.index.max().date(),
    )
if "min_gap_pts" not in st.session_state:
    st.session_state["min_gap_pts"] = 20.0
if "min_gap_pct" not in st.session_state:
    st.session_state["min_gap_pct"] = 0.5

min_date = gap_table.index.min().date()
max_date = gap_table.index.max().date()

# --- Current defaults from session state ---

date_range_default = st.session_state["date_range"]
gap_mode_default = st.session_state["gap_mode"]
min_gap_pts_default = st.session_state["min_gap_pts"]
min_gap_pct_default = st.session_state["min_gap_pct"]

gap_mode_options = ["Percent of previous RTH close", "Absolute points"]
gap_mode_index = gap_mode_options.index(gap_mode_default)

# --- Widgets (reactive: enable/disable immediately) ---

date_range = st.date_input(
    "Session date range (RTH day)",
    value=date_range_default,
    min_value=min_date,
    max_value=max_date,
)

gap_mode = st.radio(
    "Gap filter mode",
    options=gap_mode_options,
    index=gap_mode_index,
)

col1, col2 = st.columns(2)
with col1:
    min_gap_pts = st.number_input(
        "Minimum |gap| size in points",
        min_value=0.0,
        value=min_gap_pts_default,
        step=1.0,
        help="Example: 20 means |gap| >= 20 points.",
        disabled=(gap_mode == "Percent of previous RTH close"),
        key="min_gap_pts_input",
    )

with col2:
    min_gap_pct = st.number_input(
        "Minimum |gap| size in percent",
        min_value=0.0,
        value=min_gap_pct_default,
        step=0.1,
        help="Example: 0.5 means |gap| >= 0.5% of previous RTH close.",
        disabled=(gap_mode == "Absolute points"),
        key="min_gap_pct_input",
    )

apply_clicked = st.button("Apply filters")

# --- Apply filters only when button is pressed ---

if apply_clicked:
    # Normalize date_range
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date, end_date = date_range, date_range

    # Save latest widget values
    st.session_state["date_range"] = (start_date, end_date)
    st.session_state["gap_mode"] = gap_mode
    st.session_state["min_gap_pts"] = float(min_gap_pts)
    st.session_state["min_gap_pct"] = float(min_gap_pct)

    # Start from full gap_table
    mask_date = (gap_table.index.date >= start_date) & (gap_table.index.date <= end_date)
    filtered = gap_table.loc[mask_date].copy()

    # Apply gap filter
    if gap_mode == "Percent of previous RTH close":
        threshold = st.session_state["min_gap_pct"] / 100.0
        filtered = filtered[filtered["gap_abs_pct"] >= threshold]
    else:
        threshold = st.session_state["min_gap_pts"]
        filtered = filtered[filtered["gap_abs_points"] >= threshold]

    st.session_state["filtered_df"] = filtered
    st.session_state["current_idx"] = 0

# --- Use stored filtered result ---

filtered = st.session_state["filtered_df"]
gap_mode_applied = st.session_state["gap_mode"]

display_cols = [
    "prev_rth_close",
    "rth_open",
    "rth_close",
    "gap_points",
    "gap_abs_points",
    "gap_pct",
    "gap_abs_pct",
    "direction",
]

if filtered is None:
    st.info("No sessions yet. Adjust filters and click 'Apply filters'.")
elif filtered.empty:
    st.warning("Filters applied, but no sessions match the selected criteria.")
else:
    st.write(
        f"Sessions after filters: **{len(filtered)}** "
        f"(mode = {gap_mode_applied})"
    )

    # --- Single-example view + navigation ---

    current_idx = st.session_state.get("current_idx", 0)
    if current_idx < 0:
        current_idx = 0
    if current_idx > len(filtered) - 1:
        current_idx = len(filtered) - 1
    st.session_state["current_idx"] = current_idx

    current_row = filtered.iloc[current_idx:current_idx + 1]
    st.write(f"Showing example {current_idx + 1} of {len(filtered)}:")
    st.dataframe(current_row[display_cols].round(4))




# ================== Visualize current gap session ==================

st.subheader("5. Visualize current gap session")

filtered = st.session_state.get("filtered_df")

if filtered is None or filtered.empty:
    st.info("No sessions to visualize. Apply filters first.")
else:
    # Use and clamp current index
    current_idx = st.session_state.get("current_idx", 0)
    if current_idx < 0:
        current_idx = 0
    if current_idx > len(filtered) - 1:
        current_idx = len(filtered) - 1

    # Radio for chart mode
    chart_mode = st.radio(
        "Chart session view",
        options=["RTH only (prev + current)", "RTH + overnight (ETH-style)"],
        horizontal=True,
    )
    include_overnight = chart_mode == "RTH + overnight (ETH-style)"

    # ---- Navigation buttons under the chart (but we compute idx first) ----
    # We handle button clicks here so the chart/table both update in the same run.
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("◀ Previous example", use_container_width=True):
            current_idx = max(0, current_idx - 1)
    with nav_col2:
        if st.button("Next example ▶", use_container_width=True):
            current_idx = min(len(filtered) - 1, current_idx + 1)

    # Persist updated index
    st.session_state["current_idx"] = current_idx

    # Now pick session based on possibly-updated index
    session_date = filtered.index[current_idx]
    st.write(f"Visualizing session date: **{session_date.date().isoformat()}**")

    try:
        fig = build_gap_candle_figure(
            df_1m,
            gap_table,
            session_date,
            include_overnight=include_overnight,
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"scrollZoom": True},
        )
    except Exception as e:
        st.error(f"Could not build candlestick figure: {e}")

