"""
utils/dashboard_helpers.py
--------------------------
Filter application and chart-axis helpers for pages/3_Dashboard.py.

All functions are pure or near-pure (take DataFrames / plain dicts, return
data structures). This keeps the Dashboard page file focused on Streamlit
layout and makes these helpers independently testable.

Public API
----------
detect_datetime_cols(df)                       → list[str]
apply_filters(df, filters)                     → pd.DataFrame
build_kpi_cards(df_filtered, df_full, ...)     → list[dict]
get_axis_options(df, chart_type)               → dict
warn_non_time_x(col, df)                       → str | None
LARGE_DATASET_THRESHOLD                        : int  (50_000)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd


# ─── Constants ────────────────────────────────────────────────────────────────

# Datasets larger than this trigger the explicit "Apply Filters" button
# instead of reactive re-running on every widget change.
LARGE_DATASET_THRESHOLD: int = 50_000


# ─── Datetime column detection ────────────────────────────────────────────────

def detect_datetime_cols(df: pd.DataFrame) -> list:
    """
    Return the names of all columns that contain (or can be parsed as) datetimes.

    Checks native datetime dtypes first, then attempts pd.to_datetime() on
    object columns using a 50-row sample to avoid expensive full-column parsing.
    A column is accepted if ≥ 80% of the sampled values parse successfully.

    Returns
    -------
    list[str] : column names classified as datetime, in original column order
    """
    dt_cols: list = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            dt_cols.append(col)
        elif df[col].dtype == object:
            sample = df[col].dropna().head(50)
            if len(sample) == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(sample)
                if parsed.notna().sum() >= len(sample) * 0.8:
                    dt_cols.append(col)
            except (ValueError, TypeError):
                pass
    return dt_cols


# ─── Filter application ───────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply a dictionary of user-selected filters to df using AND logic.

    All active conditions must be satisfied simultaneously (intersection).

    Expected ``filters`` schema
    ---------------------------
    {
        "date_col":    str | None,          # which column holds dates
        "date_range":  (date, date) | None, # inclusive start / end
        "categorical": { col: [values] },   # allowed values per column
        "numeric":     { col: (min, max) }, # inclusive range per column
    }

    Parameters
    ----------
    df      : the full (unfiltered) DataFrame
    filters : dict matching the schema above

    Returns
    -------
    pd.DataFrame : a filtered copy; original df is never mutated
    """
    result = df.copy()

    # ── Date range filter ──────────────────────────────────────────────────────
    date_col   = filters.get("date_col")
    date_range = filters.get("date_range")
    if date_col and date_range and len(date_range) == 2:
        parsed_col = pd.to_datetime(result[date_col], errors="coerce")
        start = pd.Timestamp(date_range[0])
        end   = pd.Timestamp(date_range[1])
        result = result[(parsed_col >= start) & (parsed_col <= end)]

    # ── Categorical multi-select filters ──────────────────────────────────────
    for col, allowed_vals in (filters.get("categorical") or {}).items():
        if col in result.columns and allowed_vals is not None:
            result = result[result[col].isin(allowed_vals)]

    # ── Numeric range slider filters ──────────────────────────────────────────
    for col, bounds in (filters.get("numeric") or {}).items():
        if col in result.columns and bounds is not None:
            low, high = bounds
            result = result[(result[col] >= low) & (result[col] <= high)]

    return result


# ─── KPI card computation ─────────────────────────────────────────────────────

def build_kpi_cards(
    df_filtered: pd.DataFrame,
    df_full: pd.DataFrame,
    numeric_cols: list,
    date_col: Optional[str] = None,
) -> list:
    """
    Compute KPI card data (label, value, delta) ready for st.metric().

    Cards produced
    --------------
    1. Filtered row count vs total
    2. Sum of first numeric column  (% change vs full dataset)
    3. Mean of second numeric column (% change vs full dataset)
    4. Period-over-period trend of third numeric column
       — only when a date column exists (compares 2nd half vs 1st half of
         the filtered date range)

    Parameters
    ----------
    df_filtered  : post-filter DataFrame
    df_full      : original unfiltered DataFrame (used as baseline)
    numeric_cols : ordered list of numeric column names from df_full
    date_col     : name of the datetime column, or None

    Returns
    -------
    list[dict] : each dict has keys "label", "value", "delta" (delta may be None)
    """
    cards: list = []
    total_rows    = len(df_full)
    filtered_rows = len(df_filtered)

    # ── Card 1: Row count ──────────────────────────────────────────────────────
    pct_rows = (
        round((filtered_rows / total_rows - 1) * 100, 1) if total_rows > 0 else 0
    )
    cards.append({
        "label": "Filtered Rows",
        "value": f"{filtered_rows:,}",
        "delta": f"{pct_rows:+.1f}% vs total",
    })

    # ── Cards 2 & 3: Numeric aggregates ───────────────────────────────────────
    for i, col in enumerate(numeric_cols[:2]):
        if col not in df_filtered.columns:
            continue
        if i == 0:
            val      = df_filtered[col].sum()
            full_val = df_full[col].sum()
            label    = f"Σ {col}"
        else:
            val      = df_filtered[col].mean()
            full_val = df_full[col].mean()
            label    = f"Avg {col}"

        delta = None
        if full_val and full_val != 0:
            pct   = round((val / full_val - 1) * 100, 1)
            delta = f"{pct:+.1f}% vs total"

        cards.append({
            "label": label,
            "value": f"{val:,.2f}",
            "delta": delta,
        })

    # ── Card 4: Period-over-period trend ──────────────────────────────────────
    if date_col and len(numeric_cols) >= 3:
        col = numeric_cols[2]
        try:
            parsed_dates = pd.to_datetime(
                df_filtered[date_col], errors="coerce"
            ).dropna()
            if len(parsed_dates) > 1:
                mid = parsed_dates.min() + (parsed_dates.max() - parsed_dates.min()) / 2
                parsed_full = pd.to_datetime(df_filtered[date_col], errors="coerce")
                first_half  = df_filtered[parsed_full <= mid][col].mean()
                second_half = df_filtered[parsed_full > mid][col].mean()
                if first_half and first_half != 0 and not np.isnan(first_half):
                    trend_pct = round((second_half / first_half - 1) * 100, 1)
                    cards.append({
                        "label": f"{col} Period Trend",
                        "value": f"{second_half:,.2f}",
                        "delta": f"{trend_pct:+.1f}% (2nd half vs 1st)",
                    })
        except Exception:
            pass  # Silently skip if date parsing fails

    return cards


# ─── Axis option resolution per chart type ────────────────────────────────────

def get_axis_options(df: pd.DataFrame, chart_type: str) -> dict:
    """
    Return valid X / Y / Color axis column lists for a given chart type.

    Rules applied
    -------------
    Bar       : X = any column,      Y = numeric,   Color = categorical
    Line      : X = datetime-first,  Y = numeric,   Color = categorical
    Scatter   : X = numeric,         Y = numeric,   Color = any
    Box Plot  : X = categorical,     Y = numeric,   Color = categorical
    Heatmap   : No axis selection (uses all numeric columns automatically)

    Parameters
    ----------
    df         : the current (possibly filtered) DataFrame
    chart_type : one of "Bar", "Line", "Scatter", "Box Plot", "Heatmap"

    Returns
    -------
    dict with keys: x_options (list), y_options (list), color_options (list)
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols     = df.select_dtypes(include="object").columns.tolist()
    all_cols     = df.columns.tolist()

    # Identify datetime-parseable columns for intelligent axis ordering
    dt_cols: list = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            dt_cols.append(col)
        elif df[col].dtype == object:
            sample = df[col].dropna().head(30)
            try:
                parsed = pd.to_datetime(sample)
                if len(sample) > 0 and parsed.notna().sum() >= len(sample) * 0.8:
                    dt_cols.append(col)
            except (ValueError, TypeError):
                pass

    if chart_type == "Bar":
        return {
            "x_options":     all_cols,
            "y_options":     numeric_cols,
            "color_options": ["None"] + cat_cols,
        }
    elif chart_type == "Line":
        # Promote datetime columns to the top of the X list
        x_opts = dt_cols + [c for c in all_cols if c not in dt_cols]
        return {
            "x_options":     x_opts,
            "y_options":     numeric_cols,
            "color_options": ["None"] + cat_cols,
        }
    elif chart_type == "Scatter":
        return {
            "x_options":     numeric_cols,
            "y_options":     numeric_cols,
            "color_options": ["None"] + cat_cols + numeric_cols,
        }
    elif chart_type == "Box Plot":
        # X axis is the grouping column (optional) — "None" means one overall box
        return {
            "x_options":     ["None"] + cat_cols,
            "y_options":     numeric_cols,
            "color_options": ["None"] + cat_cols,
        }
    else:
        # Heatmap: no user axis selection needed
        return {"x_options": [], "y_options": [], "color_options": []}


# ─── Trend-axis warning ───────────────────────────────────────────────────────

def warn_non_time_x(col: str, df: pd.DataFrame) -> Optional[str]:
    """
    Return a warning string when ``col`` is unsuitable as the X axis of a
    time-series / trend chart, or None if it is appropriate.

    A column is considered time-suitable if it is:
    - A native datetime64 dtype, OR
    - An object column where ≥ 80% of a 50-row sample parses as datetime

    Parameters
    ----------
    col : column name to test
    df  : DataFrame containing the column

    Returns
    -------
    str  : warning message if col is NOT time-suitable
    None : if col is fine for a trend axis
    """
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return None  # Native datetime — perfectly suitable

    if df[col].dtype == object:
        sample = df[col].dropna().head(50)
        if len(sample) > 0:
            try:
                parsed = pd.to_datetime(sample)
                if parsed.notna().sum() >= len(sample) * 0.8:
                    return None  # Parseable as datetime — fine
            except (ValueError, TypeError):
                pass

    # Column is not datetime-like → warn the user
    return (
        f"⚠️ **'{col}'** is not a datetime column. "
        "Trend / Line charts work best with a time-based X axis. "
        "Consider switching to a datetime column, or use a Bar chart instead."
    )
