# RTH Gap & FVG Session Viewer

A free, browser-based tool to study **significant Regular Trading Hours (RTH) gaps** together with the **first Fair Value Gaps (FVGs)** that form after the 09:30 RTH open.

Instead of scrolling through days one-by-one on a charting platform, this app automatically:

- Scans historical 1-minute data for **RTH gaps**.
- Lets you **filter sessions** by date and gap size (points or %).
- Presents sessions **one at a time** with an interactive chart.
- Highlights the **first 3 FVGs** after 09:30 on the current RTH day.

> The goal is to quickly review how early FVGs behave around significant RTH gaps – functionality that’s not natively available in TradingView or FXReplay in this way.

---

## Features

### 1. Built-in historical dataset

- Uses a local CSV of **1-minute OHLCV data** (e.g. index futures).
- Loaded from `data/historical_1m.csv`.
- All preprocessing is cached for performance.

### 2. Automated RTH gap detection

For each trading day:

- Defines **previous RTH**: 09:30–16:14 (America/New_York).
- Defines **current RTH**: 09:30–16:14.
- Computes:
  - `prev_rth_close`
  - `rth_open`
  - `rth_close`
  - `gap_points` and `gap_abs_points`
  - `gap_pct` and `gap_abs_pct`
  - `direction` (up / down)

These are stored in a daily **gap table**, which drives all filtering.

### 3. Flexible session filtering

Section **4. Filter sessions by date and gap size**:

- Date range filter on the RTH session date.
- Gap mode toggle:
  - **Percent of previous RTH close**  
  - **Absolute points**
- Two input fields:
  - Min |gap| in points
  - Min |gap| in percent
- Only the relevant field is enabled based on the mode.
- **Apply filters** button:
  - Filters the precomputed gap table.
  - Stores the result in session state.
  - Resets to the first example.
- Shows **one session at a time**:
  - “Showing example X of N” + a one-row table of gap stats.

### 4. One-by-one navigation

- **Previous example** / **Next example** buttons under the chart.
- Index is stored in `st.session_state`, so:
  - The current table row (Section 4) and the chart (Section 5) stay in sync.
  - You can flip through a filtered set of sessions quickly.

### 5. RTH + FVG visualization

Section **5. Visualize current gap session**:

- Shows a candlestick chart of:
  - **RTH only mode**: previous RTH (09:30–16:14) + current RTH (09:30–16:14), back-to-back.
  - **RTH + overnight (ETH-style)**: previous RTH 09:30 → current RTH 16:14 with **all overnight bars in between** (no visual gaps).
- Candle styling:
  - Dark theme background.
  - No horizontal caps on wicks.
- Gap overlay:
  - Horizontal line at **previous RTH close** (red).
  - Horizontal line at **current RTH open** (blue).
  - Three “quadrant” levels (25/50/75%) between prev close and current open.
- **FVG detection (current RTH only)**:
  - 3-bar FVG logic:
    - Bullish: `low[bar3] > high[bar1]`
    - Bearish: `high[bar3] < low[bar1]`
  - Scans from 09:30 onwards on the current RTH day.
  - Takes the **first 3** FVGs found.
  - Each FVG is drawn as a rectangle:
    - Extends **10 minutes (10 bars)** forward.
    - Semi-transparent green (bullish) or red (bearish).
- **ETH-style extras**:
  - Includes all overnight price action between the two RTH sessions.
  - Adds a faint vertical line at the current day’s **09:30** bar to mark RTH start.

### 6. Time axis behaviour

Because we use a **category axis** (one bar per minute, back-to-back):

- There is **no horizontal gap** between 16:14 and the next day’s 09:30.
- Timestamps are thinned so they remain readable:
  - RTH-only mode → labels every **15 minutes**.
  - RTH + overnight (ETH-style) → labels every **1 hour**.

### 7. Performance

- All heavy steps are cached with `@st.cache_data`:
  - CSV load.
  - 1-minute preprocessing.
  - Gap table computation.
- Filter interactions only touch the **already-computed** gap table.
- This keeps the app responsive both locally and when deployed on free hosting.

---

## Tech stack

- **Python**
- **Streamlit**
- **pandas**, **numpy**
- **plotly** (candlestick + overlays)

