"""
utils/data_profiler.py
----------------------
Automated data-profiling engine for the AI-Powered Data Analyst Assistant.

All functions are **pure** (DataFrame in → plain Python data structures out).
This makes them independently testable and easy to explain in interviews
without needing to understand any Streamlit layout code.

Public API
----------
build_full_profile(df)              → list[dict]   one dict per column
build_quality_warnings(df, profile) → list[dict]   each has level + message
build_plain_language_summary(...)   → str           plain-English paragraph
detect_column_types(df)             → dict          col → 'numeric'|'datetime'|'categorical'
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from itertools import combinations
from typing import Optional


# ─── Column type detection ────────────────────────────────────────────────────

def detect_column_types(df: pd.DataFrame) -> dict:
    """
    Classify every column as 'numeric', 'datetime', or 'categorical'.

    Strategy
    --------
    1. Native numeric dtype  → 'numeric'
    2. Native datetime dtype → 'datetime'
    3. Object / string dtype → try parsing a 50-row sample as datetime;
       if ≥ 80% parse successfully → 'datetime', otherwise → 'categorical'
    4. Anything else (bool, category, …) → 'categorical'

    Returns
    -------
    dict : { column_name: 'numeric' | 'datetime' | 'categorical' }
    """
    col_types: dict = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_types[col] = "datetime"
        elif df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            # Sample-based datetime inference — avoids full-column overhead
            sample = df[col].dropna().head(50)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(sample)
                if parsed.notna().sum() >= len(sample) * 0.8:
                    col_types[col] = "datetime"
                else:
                    col_types[col] = "categorical"
            except (ValueError, TypeError):
                col_types[col] = "categorical"
        else:
            # bool, Categorical, etc.
            col_types[col] = "categorical"
    return col_types


# ─── Per-column profilers ─────────────────────────────────────────────────────

def profile_numeric_column(series: pd.Series) -> dict:
    """
    Compute descriptive statistics for a single numeric column.

    Outlier detection uses the **IQR fence method**:
    - Lower fence = Q1 − 1.5 × IQR
    - Upper fence = Q3 + 1.5 × IQR
    Any value outside the fences is counted as an outlier.

    Returns
    -------
    dict with keys: min, max, mean, median, std, skewness, outlier_count
    """
    clean = series.dropna()
    if len(clean) == 0:
        return {k: None for k in
                ["min", "max", "mean", "median", "std", "skewness", "outlier_count"]}

    q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outlier_count = int(((clean < lower_fence) | (clean > upper_fence)).sum())

    return {
        "min":           round(float(clean.min()), 4),
        "max":           round(float(clean.max()), 4),
        "mean":          round(float(clean.mean()), 4),
        "median":        round(float(clean.median()), 4),
        "std":           round(float(clean.std()), 4),
        "skewness":      round(float(clean.skew()), 4),
        "outlier_count": outlier_count,
    }


def profile_categorical_column(series: pd.Series, total_rows: int) -> dict:
    """
    Compute profile statistics for a categorical (object/string) column.

    Cardinality labels
    ------------------
    - 'low'    : ≤ 10 unique values
    - 'medium' : 11–50 unique values
    - 'high'   : > 50 unique values

    ID column heuristic: uniqueness ratio > 0.95 (near-100% unique values)
    suggests this column acts as an identifier rather than a true category.

    Returns
    -------
    dict with keys:
        unique_count  : int
        cardinality   : 'low' | 'medium' | 'high'
        is_id_column  : bool
        top_values    : list of (value_str, count, pct_float) for top-5 entries
    """
    clean = series.dropna()
    unique_count = int(clean.nunique())
    uniqueness_ratio = unique_count / total_rows if total_rows > 0 else 0

    if unique_count <= 10:
        cardinality = "low"
    elif unique_count <= 50:
        cardinality = "medium"
    else:
        cardinality = "high"

    is_id_column = uniqueness_ratio > 0.95

    vc = clean.value_counts().head(5)
    top_values = [
        (str(val), int(cnt), round(cnt / total_rows * 100, 1))
        for val, cnt in vc.items()
    ]

    return {
        "unique_count": unique_count,
        "cardinality":  cardinality,
        "is_id_column": is_id_column,
        "top_values":   top_values,
    }


def profile_datetime_column(series: pd.Series) -> dict:
    """
    Compute profile statistics for a datetime column.

    Gap detection
    -------------
    Converts to sorted unique **day-level** dates and counts consecutive
    pairs that are more than 1 day apart. A gap suggests missing data in
    a time series (e.g. no transactions on weekends, missing months, etc.).

    Returns
    -------
    dict with keys: min_date, max_date, span_days, gap_count
    """
    parsed = pd.to_datetime(series, errors="coerce").dropna()
    if len(parsed) == 0:
        return {"min_date": None, "max_date": None, "span_days": None, "gap_count": None}

    min_date = parsed.min()
    max_date = parsed.max()
    span_days = int((max_date - min_date).days)

    # Sort unique day-normalised dates then check consecutive diffs
    sorted_days = parsed.dt.normalize().drop_duplicates().sort_values()
    diffs = sorted_days.diff().dropna()
    gap_count = int((diffs > pd.Timedelta(days=1)).sum())

    return {
        "min_date":  str(min_date.date()),
        "max_date":  str(max_date.date()),
        "span_days": span_days,
        "gap_count": gap_count,
    }


# ─── Full profile builder ─────────────────────────────────────────────────────

def build_full_profile(df: pd.DataFrame) -> list:
    """
    Build a per-column profile for the entire DataFrame.

    Calls detect_column_types() then delegates to the appropriate
    per-column profiler for each column.

    Returns
    -------
    list of dicts, one per column, each containing:
        col_name      : str
        dtype_str     : str    (e.g. 'float64', 'object')
        missing_count : int
        missing_pct   : float  (percentage, e.g. 12.5)
        col_type      : 'numeric' | 'datetime' | 'categorical'
        numeric_stats : dict | None   (only for numeric columns)
        cat_stats     : dict | None   (only for categorical columns)
        dt_stats      : dict | None   (only for datetime columns)
    """
    total_rows = len(df)
    col_types = detect_column_types(df)
    profile = []

    for col in df.columns:
        series = df[col]
        try:
            missing_count = int(series.isnull().sum())
        except Exception:
            missing_count = 0
        missing_pct = (
            round(missing_count / total_rows * 100, 1) if total_rows > 0 else 0.0
        )
        col_type = col_types.get(col, "categorical")

        entry: dict = {
            "col_name":      col,
            "dtype_str":     str(series.dtype),
            "missing_count": missing_count,
            "missing_pct":   missing_pct,
            "col_type":      col_type,
            "numeric_stats": None,
            "cat_stats":     None,
            "dt_stats":      None,
            "profile_error": None,
        }

        try:
            if col_type == "numeric":
                entry["numeric_stats"] = profile_numeric_column(series)
            elif col_type == "datetime":
                entry["dt_stats"] = profile_datetime_column(
                    pd.to_datetime(series, errors="coerce")
                )
            else:
                # Safely cast to string before categorical profiling so
                # boolean / mixed-type columns never raise errors
                safe_series = series.astype(str).where(series.notna(), other=None)
                entry["cat_stats"] = profile_categorical_column(safe_series, total_rows)
        except Exception as exc:
            # If profiling fails for this column, mark it and continue
            entry["profile_error"] = str(exc)
            entry["col_type"] = "categorical"
            try:
                safe = series.dropna().astype(str)
                entry["cat_stats"] = {
                    "unique_count": int(safe.nunique()),
                    "cardinality":  "unknown",
                    "is_id_column": False,
                    "top_values":   [],
                }
            except Exception:
                pass

        profile.append(entry)

    return profile


# ─── Data quality warnings ─────────────────────────────────────────────────────

def build_quality_warnings(df: pd.DataFrame, profile: list) -> list:
    """
    Auto-generate data quality warnings from the computed profile.

    Checks (in order)
    -----------------
    1. Columns with > 30% missing values  → level 'error'
    2. Constant or near-constant columns  → level 'warning' / 'info'
    3. Duplicate rows                     → level 'warning'
    4. Case / abbreviation variants in categorical columns
       (e.g. 'Male', 'male', 'M') via difflib.SequenceMatcher
       — capped at 500 unique values per column for performance
    5. Highly correlated numeric pairs (|r| > 0.85) → level 'info'

    Parameters
    ----------
    df      : original (unfiltered) DataFrame
    profile : output of build_full_profile(df)

    Returns
    -------
    list of dicts : [{ "level": "error"|"warning"|"info", "message": str }]
    """
    warnings_list: list = []
    total_rows = len(df)

    # ── 1. High missing values ────────────────────────────────────────────────
    for entry in profile:
        if entry["missing_pct"] > 30:
            warnings_list.append({
                "level": "error",
                "message": (
                    f"**{entry['col_name']}** has **{entry['missing_pct']}% missing values** "
                    f"({entry['missing_count']:,} of {total_rows:,} rows). "
                    "Consider imputation or removal before modelling."
                ),
            })

    # ── 2. Constant / near-constant columns ───────────────────────────────────
    for entry in profile:
        clean = df[entry["col_name"]].dropna()
        n_unique = clean.nunique()
        if n_unique <= 1:
            warnings_list.append({
                "level": "warning",
                "message": (
                    f"**{entry['col_name']}** is **constant** ({n_unique} unique value). "
                    "It adds no analytical signal — consider dropping it."
                ),
            })
        elif n_unique == 2 and entry["col_type"] == "numeric":
            warnings_list.append({
                "level": "info",
                "message": (
                    f"**{entry['col_name']}** has only **2 unique numeric values** — "
                    "it may be better treated as a binary category (0/1 flag)."
                ),
            })

    # ── 3. Duplicate rows ────────────────────────────────────────────────────
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        dup_pct = round(dup_count / total_rows * 100, 1) if total_rows > 0 else 0
        warnings_list.append({
            "level": "warning",
            "message": (
                f"**{dup_count:,} duplicate rows** detected "
                f"({dup_pct}% of dataset). "
                "Duplicates can skew aggregations and model training."
            ),
        })

    # ── 4. Categorical value variant detection (fuzzy matching) ───────────────
    for entry in profile:
        if entry["col_type"] != "categorical" or entry.get("cat_stats") is None:
            continue
        # Skip ID-like columns — too many values, comparisons would be meaningless
        if entry["cat_stats"]["is_id_column"]:
            continue

        series = df[entry["col_name"]].dropna()
        # Cap at 500 unique values to keep O(n²) comparisons manageable
        unique_vals = [str(v) for v in series.unique()][:500]

        suspect_groups: list = []
        checked: set = set()

        for i, v1 in enumerate(unique_vals):
            group = [v1]
            for v2 in unique_vals[i + 1:]:
                pair = tuple(sorted([v1, v2]))
                if pair in checked:
                    continue
                checked.add(pair)

                v1_norm = v1.strip().lower()
                v2_norm = v2.strip().lower()

                # Fast path: exact match after stripping and lowercasing
                if v1_norm == v2_norm:
                    group.append(v2)
                    continue

                # Short value abbreviation check (e.g. 'M' vs 'Male')
                if len(v1) <= 3 or len(v2) <= 3:
                    if v1_norm[:1] == v2_norm[:1]:
                        ratio = SequenceMatcher(None, v1_norm, v2_norm).ratio()
                        if ratio > 0.6:
                            group.append(v2)
                else:
                    # Full fuzzy similarity
                    ratio = SequenceMatcher(None, v1_norm, v2_norm).ratio()
                    if ratio > 0.85:
                        group.append(v2)

            if len(group) > 1 and group not in [g for g in suspect_groups]:
                suspect_groups.append(group)

        if suspect_groups:
            # Show up to 3 example groups in the warning message
            examples = "; ".join(
                ["/".join(g) for g in suspect_groups[:3]]
            )
            warnings_list.append({
                "level": "warning",
                "message": (
                    f"**{entry['col_name']}** may have **inconsistent value variants**: "
                    f"{examples}. Consider standardising before grouping."
                ),
            })

    # ── 5. Highly correlated numeric pairs ───────────────────────────────────
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] >= 2:
        corr_matrix = numeric_df.corr().abs()
        for col1, col2 in combinations(numeric_df.columns, 2):
            r = corr_matrix.loc[col1, col2]
            if r > 0.85:
                warnings_list.append({
                    "level": "info",
                    "message": (
                        f"**{col1}** and **{col2}** are **highly correlated** "
                        f"(r = {r:.2f}). One may be redundant — "
                        "check for multicollinearity before building regression models."
                    ),
                })

    return warnings_list


# ─── Plain-language summary ───────────────────────────────────────────────────

def build_plain_language_summary(
    df: pd.DataFrame,
    profile: list,
    warnings: list,
) -> str:
    """
    Generate a plain-English summary paragraph from computed statistics.

    Every claim is directly traceable to a number in the profile —
    this is NOT an LLM-generated paragraph, so the summary is always
    factually grounded.

    Returns
    -------
    str : a 4–8 sentence paragraph suitable for st.info() on the EDA page
          and as structured context for the AI Insights prompt.
    """
    total_rows, total_cols = df.shape
    total_missing = sum(e["missing_count"] for e in profile)
    cell_count = total_rows * total_cols
    missing_pct = round(total_missing / cell_count * 100, 1) if cell_count > 0 else 0

    high_missing_cols = [e["col_name"] for e in profile if e["missing_pct"] > 30]
    dup_count = int(df.duplicated().sum())

    numeric_entries  = [e for e in profile if e["col_type"] == "numeric"]
    cat_entries      = [e for e in profile if e["col_type"] == "categorical"]
    dt_entries       = [e for e in profile if e["col_type"] == "datetime"]

    lines: list = []

    # Sentence 1: shape + type breakdown
    type_parts = [f"{len(numeric_entries)} numeric", f"{len(cat_entries)} categorical"]
    if dt_entries:
        type_parts.append(f"{len(dt_entries)} datetime")
    lines.append(
        f"This dataset has **{total_rows:,} rows** and **{total_cols} columns** "
        f"({', '.join(type_parts)})."
    )

    # Sentence 2: missing values
    if total_missing == 0:
        lines.append("There are **no missing values** — the dataset is complete.")
    elif high_missing_cols:
        lines.append(
            f"Overall missing data is **{missing_pct}%**. "
            f"{len(high_missing_cols)} column(s) exceed 30% missing: "
            + ", ".join(f"**{c}**" for c in high_missing_cols) + "."
        )
    else:
        lines.append(
            f"Overall missing data is low at **{missing_pct}%** — "
            "no column exceeds the 30% threshold."
        )

    # Sentence 3: duplicates
    if dup_count > 0:
        lines.append(
            f"**{dup_count:,} duplicate rows** were found "
            f"({round(dup_count / total_rows * 100, 1)}% of total)."
        )

    # Sentences 4–5: notable numeric columns (up to 2)
    for entry in numeric_entries[:2]:
        stats = entry.get("numeric_stats") or {}
        if stats.get("outlier_count") is not None:
            skew_note = ""
            skew = stats.get("skewness") or 0
            if abs(skew) > 1:
                skew_note = f", {'right' if skew > 0 else 'left'}-skewed (skewness={skew})"
            lines.append(
                f"**{entry['col_name']}** ranges from {stats['min']} to {stats['max']}"
                f"{skew_note}, with **{stats['outlier_count']} outlier(s)** detected by IQR."
            )

    # Sentences 6–7: notable categorical columns (up to 2)
    for entry in cat_entries[:2]:
        stats = entry.get("cat_stats") or {}
        if stats:
            if stats["is_id_column"]:
                lines.append(
                    f"**{entry['col_name']}** appears to be an **ID column** "
                    f"({stats['unique_count']:,} unique values, near-100% unique)."
                )
            else:
                top_vals = stats.get("top_values", [])
                top_str = ", ".join(f"{v} ({p}%)" for v, _, p in top_vals[:3]) if top_vals else ""
                lines.append(
                    f"**{entry['col_name']}** has {stats['unique_count']} unique values "
                    f"(cardinality: {stats['cardinality']})"
                    + (f". Top values: {top_str}." if top_str else ".")
                )

    # Final sentence: quality check summary
    error_count = sum(1 for w in warnings if w["level"] == "error")
    warn_count  = sum(1 for w in warnings if w["level"] == "warning")
    if error_count + warn_count > 0:
        lines.append(
            f"Quality checks flagged **{error_count} critical issue(s)** and "
            f"**{warn_count} warning(s)** — review the Data Quality section below."
        )
    else:
        lines.append("No critical data quality issues were detected.")

    return " ".join(lines)
