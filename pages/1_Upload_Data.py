"""
pages/1_Upload_Data.py
----------------------
Robust dataset uploader that handles virtually every real-world CSV/Excel file:

Encoding   : utf-8, utf-8-sig, cp1252, latin-1, iso-8859-1, cp1250, cp1251,
             big5, gb2312, shift-jis (auto-detected by trial)
Delimiters : comma, semicolon, tab, pipe, space (auto-detected via csv.Sniffer)
Excel      : .xlsx, .xls, .xlsm — multi-sheet support with sheet picker
Data clean : strips whitespace from column names, de-duplicates columns,
             drops fully-empty rows/cols (optional), resets index
Edge cases : empty file, single column, all-NaN column, duplicate headers,
             mixed-type numeric columns stored as strings
"""

import csv
import io
import os
import sys

import pandas as pd
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Upload Data · Data Analyst",
    page_icon="📂",
    layout="wide",
)

import sys, os
if os.path.dirname(__file__) + "/.." not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__) + "/..")
from utils.ui import inject_custom_css, sidebar_brand
inject_custom_css()
sidebar_brand()

st.title("📂 Upload Dataset")
st.markdown("Supports **CSV**, **TSV**, **Excel (.xlsx / .xls)** — any encoding, any delimiter.")

# ── Encoding list (ordered: most-common first) ───────────────────────────────
ENCODINGS = [
    "utf-8", "utf-8-sig",
    "cp1252",      # Windows Western (curly quotes, €, …)
    "latin-1",     # ISO-8859-1  — never raises UnicodeDecodeError
    "iso-8859-1",
    "cp1250",      # Windows Central European
    "cp1251",      # Windows Cyrillic
    "big5",        # Traditional Chinese
    "gb2312",      # Simplified Chinese
    "shift-jis",   # Japanese
]


def detect_delimiter(raw_bytes: bytes, encoding: str) -> str:
    """
    Use csv.Sniffer to detect the delimiter of a CSV file.
    Falls back to comma if detection fails.
    """
    try:
        sample = raw_bytes[:4096].decode(encoding, errors="replace")
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        return ","


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a freshly-loaded DataFrame so downstream pages never
    encounter common structural problems:

    1. Strip leading/trailing whitespace from column names
    2. Replace blank / NaN column names with 'Column_N'
    3. De-duplicate column names by appending _1, _2, …
    4. Strip string columns of leading/trailing whitespace
    5. Reset the integer index
    """
    # 1 & 2 — clean column names
    new_cols = []
    for i, col in enumerate(df.columns):
        c = str(col).strip()
        if c in ("", "nan", "None"):
            c = f"Column_{i+1}"
        new_cols.append(c)

    # 3 — de-duplicate
    seen: dict = {}
    deduped = []
    for c in new_cols:
        if c in seen:
            seen[c] += 1
            deduped.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            deduped.append(c)
    df.columns = deduped

    # 4 — strip object columns
    for col in df.select_dtypes(include="object").columns:
        try:
            df[col] = df[col].str.strip()
        except Exception:
            pass

    # 5 — reset index
    df = df.reset_index(drop=True)
    return df


def try_numeric_coercion(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every object column, attempt to coerce it to numeric.
    If ≥ 80 % of non-null values parse successfully, convert the column.
    This fixes CSVs where numbers are stored as strings (e.g. "1,234.56").
    """
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna()
        if len(sample) == 0:
            continue
        # Remove common thousands separators before testing
        cleaned = sample.str.replace(",", "", regex=False).str.strip()
        coerced = pd.to_numeric(cleaned, errors="coerce")
        ratio = coerced.notna().sum() / len(sample)
        if ratio >= 0.8:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce",
            )
    return df


def read_csv_robust(file_obj) -> tuple[pd.DataFrame, dict]:
    """
    Read a CSV file trying every encoding and auto-detecting the delimiter.
    Returns (DataFrame, info_dict).
    Raises RuntimeError if nothing works.
    """
    raw = file_obj.read()
    info = {"encoding": None, "delimiter": None, "warnings": []}
    last_err = None

    for enc in ENCODINGS:
        delim = detect_delimiter(raw, enc)
        try:
            df = pd.read_csv(
                io.BytesIO(raw),
                encoding=enc,
                sep=delim,
                on_bad_lines="warn",     # skip malformed rows, don't crash
                low_memory=False,        # avoid mixed-type warnings
            )
            info["encoding"] = enc
            info["delimiter"] = delim
            if enc not in ("utf-8", "utf-8-sig"):
                info["warnings"].append(
                    f"File is **{enc}** encoded (not UTF-8). Read successfully."
                )
            if delim != ",":
                delim_name = {"\t": "Tab", ";": "Semicolon", "|": "Pipe"}.get(delim, repr(delim))
                info["warnings"].append(
                    f"Detected **{delim_name}** as delimiter (not comma)."
                )
            return df, info
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"Could not read the file with any supported encoding.\nLast error: {last_err}"
    )


def read_excel_robust(file_obj) -> tuple[pd.DataFrame, dict]:
    """Read an Excel file; if multi-sheet, let the user choose a sheet."""
    info = {"encoding": "N/A (Excel binary)", "delimiter": "N/A", "warnings": []}
    try:
        xf = pd.ExcelFile(file_obj)
        sheets = xf.sheet_names

        if len(sheets) == 1:
            df = xf.parse(sheets[0])
        else:
            chosen = st.selectbox(
                f"This Excel file has **{len(sheets)} sheets** — choose one to load:",
                options=sheets,
            )
            df = xf.parse(chosen)
            info["warnings"].append(f"Loaded sheet: **{chosen}**")

        return df, info
    except Exception as e:
        raise RuntimeError(f"Could not read the Excel file: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ════════════════════════════════════════════════════════════════════════════════

uploaded_file = st.file_uploader(
    "Upload CSV, TSV, or Excel File",
    type=["csv", "tsv", "xlsx", "xls", "xlsm"],
    help="Any encoding (UTF-8, Latin-1, Windows-1252, etc.) and any delimiter are supported.",
)

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()
    df = None
    file_info = {}

    with st.spinner("Reading file…"):
        try:
            if file_name.endswith((".xlsx", ".xls", ".xlsm")):
                df, file_info = read_excel_robust(uploaded_file)
            else:
                # Treat .csv, .tsv, and anything else as delimited text
                df, file_info = read_csv_robust(uploaded_file)
        except RuntimeError as e:
            st.error(str(e))
            st.markdown(
                "> **Tip:** Open the file in Excel or a text editor, then re-save as "
                "*CSV UTF-8 (with BOM)* to guarantee compatibility."
            )
            st.stop()

    # ── Post-load validation ─────────────────────────────────────────────────
    if df is None or df.empty:
        st.error("❌ The uploaded file appears to be empty. Please check the file and try again.")
        st.stop()

    if df.shape[1] == 1 and df.columns[0].count(",") > 2:
        st.warning(
            "⚠️ Only 1 column was detected. The file may have been read with the wrong delimiter. "
            "Try renaming it to `.tsv` if it uses tab separation."
        )

    # ── Clean the DataFrame ──────────────────────────────────────────────────
    df = clean_dataframe(df)
    df = try_numeric_coercion(df)

    # ── Drop fully-empty rows / columns ─────────────────────────────────────
    before_rows = len(df)
    df.dropna(how="all", inplace=True)         # rows where every cell is NaN
    dropped_rows = before_rows - len(df)

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df.drop(columns=empty_cols, inplace=True)
        file_info.setdefault("warnings", []).append(
            f"Dropped **{len(empty_cols)}** fully-empty column(s): {', '.join(empty_cols)}"
        )
    if dropped_rows:
        file_info.setdefault("warnings", []).append(
            f"Dropped **{dropped_rows}** fully-empty row(s)."
        )

    df = df.reset_index(drop=True)

    import uuid
    st.session_state["dataset_id"] = str(uuid.uuid4())
    st.session_state["dataset_filename"] = uploaded_file.name
    st.session_state["df"] = df
    # Clear stale cached profile so EDA page recomputes for new data
    for key in ["dataset_profile_summary", "dataset_profile"]:
        st.session_state.pop(key, None)

    # ── Success banner ───────────────────────────────────────────────────────
    st.success(f"✅ **{uploaded_file.name}** loaded successfully!")

    # Show any soft warnings (encoding, delimiter, dropped rows/cols)
    for w in file_info.get("warnings", []):
        st.info(f"ℹ️ {w}")

    # ── Dataset Metrics ──────────────────────────────────────────────────────
    st.subheader("📊 Dataset at a Glance")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📄 Rows",    f"{df.shape[0]:,}")
    c2.metric("🗂️ Columns", f"{df.shape[1]:,}")
    c3.metric("🔢 Numeric cols",    str(len(df.select_dtypes(include="number").columns)))
    c4.metric("🔡 Text cols",       str(len(df.select_dtypes(include="object").columns)))
    c5.metric("❓ Missing values",  f"{int(df.isnull().sum().sum()):,}")

    # ── File info ────────────────────────────────────────────────────────────
    with st.expander("🔍 File read details"):
        st.markdown(f"- **Encoding detected:** `{file_info.get('encoding', 'N/A')}`")
        st.markdown(f"- **Delimiter detected:** `{repr(file_info.get('delimiter', 'N/A'))}`")
        st.markdown(f"- **File size:** {uploaded_file.size / 1024:.1f} KB")

    # ── Preview ──────────────────────────────────────────────────────────────
    st.subheader("📋 Data Preview (first 100 rows)")
    st.dataframe(df.head(100), use_container_width=True)

    # ── Column Information ───────────────────────────────────────────────────
    st.subheader("📝 Column Summary")

    col_info = pd.DataFrame({
        "Column":          df.columns,
        "Dtype":           df.dtypes.astype(str).values,
        "Non-Null Count":  df.notnull().sum().values,
        "Null Count":      df.isnull().sum().values,
        "Null %":          (df.isnull().mean() * 100).round(1).astype(str) + "%",
        "Unique Values":   [df[c].nunique() for c in df.columns],
        "Sample Value":    [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—"
                            for c in df.columns],
    })
    st.dataframe(col_info, use_container_width=True, hide_index=True)

    st.info("👈 Navigate to **Data Analysis**, **Dashboard**, or **AI Insights** from the sidebar.")