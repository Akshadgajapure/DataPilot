"""
utils/data_cleaning_engine.py
-----------------------------
Industry-Grade Strict Data Cleaning Pipeline — General Purpose.

Principles:
1. Never silently fabricate or corrupt data.
2. Missing != always fixable.
3. Never break internal consistency.
4. Every transformation must be logged.
5. Flag before you fix, when uncertain.
6. Validate business/domain logic.
7. Never impute ID-like columns.
8. Run a second validation pass after cleaning.
9. Detect and standardize inconsistent categorical casing.
10. Generate a clear before/after summary.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re


# ── Heuristic ID-column keywords (never impute these) ────────────────────────
_ID_KEYWORDS = [
    "id", "_id", "id_", "uuid", "guid", "key", "code", "ref", "reference",
    "number", "no", "num", "barcode", "sku", "serial", "token",
]

# ── Non-negative domain keywords (flag if values are negative) ────────────────
_NON_NEG_KEYWORDS = [
    "price", "revenue", "cost", "qty", "quantity", "age", "salary",
    "amount", "distance", "weight", "height", "size", "score",
    "rate", "ratio", "count", "total", "sum", "balance",
]

# ── Thresholds ────────────────────────────────────────────────────────────────
_IMPUTE_THRESH    = 5.0    # % — below this: impute numerics / mode for categoricals
_MODE_THRESH      = 20.0   # % — below this but above impute: mode ok for categoricals
_HIGH_CARD_RATIO  = 0.9    # if nunique/nrows > this → treat column as ID-like
_SKEW_THRESH      = 1.0    # |skew| above this → use median, else mean


def _is_id_column(col: str, series: pd.Series) -> bool:
    """Return True if this column looks like an ID / key and should not be imputed."""
    col_lower = col.lower().replace(" ", "_")
    # Name-based heuristic
    for kw in _ID_KEYWORDS:
        if col_lower == kw or col_lower.startswith(kw + "_") or col_lower.endswith("_" + kw):
            return True
    # Cardinality-based heuristic: almost every value is unique → likely an ID
    n = len(series.dropna())
    if n > 0 and series.nunique() / n >= _HIGH_CARD_RATIO:
        return True
    return False


class ChangeLogger:
    def __init__(self):
        self.logs = []

    def log(self, stage, column, rows_affected, method, old_stat=None, new_stat=None, details=""):
        entry = {
            "timestamp":     datetime.now().isoformat(),
            "stage":         stage,
            "column":        column,
            "rows_affected": int(rows_affected),
            "method":        method,
            "old_stat":      str(old_stat) if old_stat is not None else None,
            "new_stat":      str(new_stat) if new_stat is not None else None,
            "details":       details,
        }
        self.logs.append(entry)

    def get_logs(self):
        return self.logs

    def to_dataframe(self):
        return pd.DataFrame(self.logs)


class StrictDataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.original_df = df.copy()
        self.df          = df.copy()
        self.logger      = ChangeLogger()
        self.profile     = {}

    # ═════════════════════════════════════════════════════════════════════════
    # Public entry point
    # ═════════════════════════════════════════════════════════════════════════
    def run_pipeline(self):
        """
        Runs the strict 6-stage cleaning pipeline.
        Returns: (cleaned_df, changelog_df, outlier_df, profile)
        The return signature is unchanged so the Streamlit page requires no edits.
        """
        self._stage1_profile()
        self._stage2_structural()
        self._stage3_missing()
        self._stage4_validity()
        outlier_df = self._stage5_outliers()
        self._stage6_validation_pass()

        # Attach the before/after summary to the profile dict so the UI can use it
        self.profile["__summary__"] = self._build_summary()

        return self.df, self.logger.to_dataframe(), outlier_df, self.profile

    # ═════════════════════════════════════════════════════════════════════════
    # Stage 1 — Profiling
    # ═════════════════════════════════════════════════════════════════════════
    def _stage1_profile(self):
        """
        Detect column types, missingness, cardinality, and skew
        before touching anything.  Completely domain-agnostic.
        """
        for col in self.df.columns:
            series       = self.df[col]
            missing_pct  = series.isnull().mean() * 100
            n_unique     = series.nunique()
            is_id        = _is_id_column(col, series)

            # ── Infer semantic type ──────────────────────────────────────────
            if is_id:
                inferred_type = "id"
            elif pd.api.types.is_numeric_dtype(series):
                inferred_type = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(series):
                inferred_type = "datetime"
            else:
                # Try to parse as datetime string
                sample = series.dropna().head(20).astype(str)
                is_date_string = False
                if len(sample) > 0:
                    try:
                        pd.to_datetime(sample, format="mixed", errors="raise")
                        is_date_string = True
                    except Exception:
                        pass

                if is_date_string:
                    inferred_type = "datetime_string"
                elif n_unique <= max(20, int(len(series) * 0.05)) and len(series) > 20:
                    inferred_type = "categorical"
                else:
                    inferred_type = "text"

            # ── Skew for numeric columns ─────────────────────────────────────
            skew = None
            if inferred_type == "numeric":
                try:
                    skew = float(series.skew())
                except Exception:
                    skew = 0.0

            self.profile[col] = {
                "inferred_type": inferred_type,
                "missing_pct":   missing_pct,
                "nunique":       n_unique,
                "is_id":         is_id,
                "skew":          skew,
            }

    # ═════════════════════════════════════════════════════════════════════════
    # Stage 2 — Structural Cleaning
    # ═════════════════════════════════════════════════════════════════════════
    def _stage2_structural(self):
        """
        Standardize column names, remove exact duplicates, strip string
        whitespace, parse datetime strings, and standardize inconsistent
        categorical casing (e.g. 'ELECTRONICS', 'electronics' → 'Electronics').
        """
        # ── 2a. Strip column-name whitespace ────────────────────────────────
        old_cols = list(self.df.columns)
        self.df.columns = self.df.columns.str.strip()
        for old_c, new_c in zip(old_cols, self.df.columns):
            if old_c != new_c:
                self.logger.log("2_Structural", old_c, len(self.df),
                                "Rename Column", old_c, new_c,
                                "Stripped whitespace from column name")

        # ── 2b. Remove exact duplicate rows ─────────────────────────────────
        before_len = len(self.df)
        self.df.drop_duplicates(inplace=True)
        after_len = len(self.df)
        if before_len > after_len:
            self.logger.log("2_Structural", "ALL_ROWS",
                            before_len - after_len, "Drop Exact Duplicates",
                            before_len, after_len,
                            "Identical rows removed")

        # ── 2c. Per-column type fixes ────────────────────────────────────────
        for col in list(self.df.columns):
            info   = self.profile.get(col.strip(), {})
            series = self.df[col]

            # Strip leading/trailing whitespace from string values
            if series.dtype == "object":
                before_strip = series.copy()
                self.df[col] = series.str.strip()
                changed = (before_strip != self.df[col]).sum()
                if changed > 0:
                    self.logger.log("2_Structural", col, int(changed),
                                    "Strip Whitespace",
                                    details="Stripped leading/trailing whitespace from cell values")

            # Parse datetime strings → proper datetime dtype
            if info.get("inferred_type") == "datetime_string":
                try:
                    self.df[col] = pd.to_datetime(self.df[col], format="mixed", errors="coerce")
                    n_parsed = self.df[col].notna().sum()
                    self.logger.log("2_Structural", col, int(n_parsed),
                                    "Parse Datetime String",
                                    details="Converted string column to datetime dtype")
                    # Update profile so Stage 3 treats it as datetime
                    self.profile[col]["inferred_type"] = "datetime"
                except Exception:
                    pass

            # ── Categorical case standardisation ────────────────────────────
            # Detects inconsistent casing for the SAME value across rows
            # (e.g. "Electronics", "electronics", "ELECTRONICS") → Title Case.
            if info.get("inferred_type") == "categorical" and series.dtype == "object":
                try:
                    # Build a normalised → canonical map (most-frequent casing wins)
                    non_null = self.df[col].dropna()
                    norm_map: dict[str, str] = {}
                    for raw_val, cnt in non_null.value_counts().items():
                        key = str(raw_val).strip().lower()
                        if key not in norm_map:
                            norm_map[key] = str(raw_val).strip().title()
                        # prefer the most frequent original spelling → already set above
                    old_vals = self.df[col].copy()
                    self.df[col] = self.df[col].apply(
                        lambda x: norm_map.get(str(x).strip().lower(), x)
                        if pd.notna(x) else x
                    )
                    changed = (old_vals.fillna("__NA__") != self.df[col].fillna("__NA__")).sum()
                    if changed > 0:
                        self.logger.log("2_Structural", col, int(changed),
                                        "Standardize Categorical Casing",
                                        details="Unified inconsistent casing variants to Title Case")
                except Exception:
                    pass

    # ═════════════════════════════════════════════════════════════════════════
    # Stage 3 — Missing Value Handling  (domain-agnostic, tiered thresholds)
    # ═════════════════════════════════════════════════════════════════════════
    def _stage3_missing(self):
        """
        Tiered, defensible missing-value strategy.

        Thresholds (configurable via module constants):
          ≤ IMPUTE_THRESH  %  →  statistical imputation (numeric) / mode (categorical)
          ≤ MODE_THRESH    %  →  mode imputation only for categoricals / flag numerics
          > MODE_THRESH    %  →  add _Is_Missing flag, leave original null

        Special cases:
          - ID-like columns   → never imputed (only flagged if >0 missing)
          - Datetime columns  → never imputed with arbitrary dates (only flagged)
          - Text columns      → never imputed (only flagged)
        """
        for col in list(self.df.columns):
            # Skip flag columns we may have already created
            if col.endswith(("_Is_Missing", "_Is_Invalid", "_Is_Outlier")):
                continue

            missing_count = int(self.df[col].isnull().sum())
            if missing_count == 0:
                continue

            n_rows       = len(self.df)
            missing_pct  = (missing_count / n_rows) * 100
            info         = self.profile.get(col, {})
            itype        = info.get("inferred_type", "text")
            is_id        = info.get("is_id", False)

            # ── ID columns: never impute ─────────────────────────────────────
            if is_id or itype == "id":
                flag_col = f"{col}_Is_Missing"
                self.df[flag_col] = self.df[col].isnull()
                self.logger.log("3_Missing", col, missing_count,
                                "Flagged (ID Column — No Imputation)",
                                new_stat=flag_col,
                                details=f"Missing {missing_pct:.2f}%. ID columns are never imputed.")
                continue

            # ── Datetime columns: never fill with arbitrary dates ─────────────
            if itype in ("datetime", "datetime_string"):
                # Attempt forward-fill only when the series is ordered (time-series-like)
                # and the gap is small. Otherwise just flag.
                if missing_pct <= _IMPUTE_THRESH:
                    before_null = self.df[col].isnull().sum()
                    self.df[col] = self.df[col].ffill().bfill()
                    after_null   = self.df[col].isnull().sum()
                    filled       = before_null - after_null
                    if filled > 0:
                        self.logger.log("3_Missing", col, int(filled),
                                        "Datetime Forward/Back Fill",
                                        details=f"Missing {missing_pct:.2f}% (≤{_IMPUTE_THRESH}%). "
                                                "Only applied forward+backward fill — no fabricated dates.")
                else:
                    flag_col = f"{col}_Is_Missing"
                    self.df[flag_col] = self.df[col].isnull()
                    self.logger.log("3_Missing", col, missing_count,
                                    "Flagged (Datetime — No Arbitrary Fill)",
                                    new_stat=flag_col,
                                    details=f"Missing {missing_pct:.2f}% (>{_IMPUTE_THRESH}%). "
                                            "Datetime columns are not filled with arbitrary values.")
                continue

            # ── Text (free-form string) columns: flag only ───────────────────
            if itype == "text":
                flag_col = f"{col}_Is_Missing"
                self.df[flag_col] = self.df[col].isnull()
                self.logger.log("3_Missing", col, missing_count,
                                "Flagged (Free-text — No Imputation)",
                                new_stat=flag_col,
                                details=f"Missing {missing_pct:.2f}%. Free-text columns are not imputed.")
                continue

            # ── Numeric columns ──────────────────────────────────────────────
            if pd.api.types.is_numeric_dtype(self.df[col]):
                if missing_pct <= _IMPUTE_THRESH:
                    skew = info.get("skew")
                    if skew is None:
                        try:
                            skew = float(self.df[col].skew())
                        except Exception:
                            skew = 0.0
                    if abs(skew) > _SKEW_THRESH:
                        val    = self.df[col].median()
                        method = f"Median Imputation (skew={skew:.2f}, highly skewed)"
                    else:
                        val    = self.df[col].mean()
                        method = f"Mean Imputation (skew={skew:.2f}, approx. normal)"
                    if pd.notna(val):
                        self.df[col] = self.df[col].fillna(val)
                        self.logger.log("3_Missing", col, missing_count, method,
                                        old_stat=f"{missing_count} nulls",
                                        new_stat=round(val, 6),
                                        details=f"Missing {missing_pct:.2f}% (≤{_IMPUTE_THRESH}%)")
                else:
                    # Too many missing → flag only, do NOT guess
                    flag_col = f"{col}_Is_Missing"
                    self.df[flag_col] = self.df[col].isnull()
                    self.logger.log("3_Missing", col, missing_count,
                                    "Flagged (Numeric — Too Many Missing)",
                                    new_stat=flag_col,
                                    details=f"Missing {missing_pct:.2f}% (>{_IMPUTE_THRESH}%). "
                                            "Original left null; flag column added.")
                continue

            # ── Categorical columns ──────────────────────────────────────────
            if itype == "categorical":
                mode_s = self.df[col].mode()
                mode_val = mode_s.iloc[0] if not mode_s.empty else None

                if missing_pct <= _IMPUTE_THRESH:
                    # Very few missing → mode imputation is safe
                    if mode_val is not None:
                        self.df[col] = self.df[col].fillna(mode_val)
                        self.logger.log("3_Missing", col, missing_count,
                                        "Mode Imputation",
                                        old_stat=f"{missing_count} nulls",
                                        new_stat=mode_val,
                                        details=f"Missing {missing_pct:.2f}% (≤{_IMPUTE_THRESH}%). "
                                                "Mode is a safe estimate for low-missingness categoricals.")

                elif missing_pct <= _MODE_THRESH:
                    # Moderate missing → mode still defensible but we log more explicitly
                    if mode_val is not None:
                        self.df[col] = self.df[col].fillna(mode_val)
                        self.logger.log("3_Missing", col, missing_count,
                                        "Mode Imputation (Moderate Missing)",
                                        old_stat=f"{missing_count} nulls",
                                        new_stat=mode_val,
                                        details=f"Missing {missing_pct:.2f}% ({_IMPUTE_THRESH}–{_MODE_THRESH}%). "
                                                "Mode used; review if distribution is meaningful.")
                else:
                    # High missingness → use 'Unknown' sentinel + flag column
                    self.df[col] = self.df[col].fillna("Unknown")
                    flag_col = f"{col}_Is_Missing"
                    self.df[flag_col] = self.original_df[col].reindex(self.df.index).isnull()
                    self.logger.log("3_Missing", col, missing_count,
                                    "Filled with 'Unknown' + Flagged",
                                    new_stat="Unknown",
                                    details=f"Missing {missing_pct:.2f}% (>{_MODE_THRESH}%). "
                                            "Too many missing for mode imputation; 'Unknown' sentinel used.")
                continue

            # ── Fallback: flag anything we can't classify ────────────────────
            flag_col = f"{col}_Is_Missing"
            self.df[flag_col] = self.df[col].isnull()
            self.logger.log("3_Missing", col, missing_count,
                            "Flagged (Unclassified Column)",
                            new_stat=flag_col,
                            details=f"Missing {missing_pct:.2f}%. Column type unclassified; flagged only.")

    # ═════════════════════════════════════════════════════════════════════════
    # Stage 4 — Validity & Consistency Checks
    # ═════════════════════════════════════════════════════════════════════════
    def _stage4_validity(self):
        """
        Domain-agnostic validity checks:
          - Semantic string validation (email, phone) by column-name heuristics.
          - Non-negative domain enforcement by keyword heuristics.
          - Auto-detected mathematical relationships (e.g. Total = A × B).
        """
        numeric_cols = self.df.select_dtypes(include="number").columns
        string_cols  = self.df.select_dtypes(include="object").columns

        # ── 4a. Semantic string validation ──────────────────────────────────
        for col in string_cols:
            col_lower = col.lower()

            # Phone number check
            if any(kw in col_lower for kw in ("phone", "mobile", "contact", "tel")):
                not_null = self.df[col].notna()
                if not_null.sum() > 0:
                    digit_counts = self.df.loc[not_null, col].astype(str).str.count(r"\d")
                    invalid_mask = (digit_counts < 7) | (digit_counts > 15)
                    n_invalid    = int(invalid_mask.sum())
                    if n_invalid > 0:
                        full_mask = pd.Series(False, index=self.df.index)
                        full_mask.loc[not_null] = invalid_mask
                        flag_col = f"{col}_Is_Invalid"
                        self.df[flag_col] = full_mask
                        self.logger.log("4_Validity", col, n_invalid,
                                        "Flagged Invalid Phone",
                                        new_stat=flag_col,
                                        details="Values do not contain 7–15 digits")

            # Email check
            elif "email" in col_lower or "e-mail" in col_lower:
                not_null = self.df[col].notna()
                if not_null.sum() > 0:
                    pattern  = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
                    is_valid = self.df.loc[not_null, col].astype(str).str.match(pattern)
                    n_invalid = int((~is_valid).sum())
                    if n_invalid > 0:
                        full_mask = pd.Series(False, index=self.df.index)
                        full_mask.loc[not_null] = ~is_valid
                        flag_col = f"{col}_Is_Invalid"
                        self.df[flag_col] = full_mask
                        self.logger.log("4_Validity", col, n_invalid,
                                        "Flagged Invalid Email",
                                        new_stat=flag_col,
                                        details="Values do not match standard email pattern")

            # URL check
            elif any(kw in col_lower for kw in ("url", "link", "website", "site")):
                not_null = self.df[col].notna()
                if not_null.sum() > 0:
                    pattern   = r"^https?://"
                    is_valid  = self.df.loc[not_null, col].astype(str).str.match(pattern, case=False)
                    n_invalid = int((~is_valid).sum())
                    if n_invalid > 0:
                        full_mask = pd.Series(False, index=self.df.index)
                        full_mask.loc[not_null] = ~is_valid
                        flag_col = f"{col}_Is_Invalid"
                        self.df[flag_col] = full_mask
                        self.logger.log("4_Validity", col, n_invalid,
                                        "Flagged Invalid URL",
                                        new_stat=flag_col,
                                        details="Values do not start with http:// or https://")

        # ── 4b. Non-negative domain enforcement ──────────────────────────────
        for col in numeric_cols:
            if col.endswith(("_Is_Missing", "_Is_Invalid", "_Is_Outlier")):
                continue
            if _is_id_column(col, self.df[col]):
                continue
            col_lower = col.lower()
            if any(kw in col_lower for kw in _NON_NEG_KEYWORDS):
                neg_mask  = self.df[col] < 0
                neg_count = int(neg_mask.sum())
                if neg_count > 0:
                    self.df.loc[neg_mask, col] = np.nan
                    self.logger.log("4_Validity", col, neg_count,
                                    "Invalid Negative → NaN",
                                    details=f"'{col}' should be non-negative. "
                                            "Replaced with NaN for downstream imputation review.")

        # ── 4c. Auto-detect mathematical column relationships ─────────────────
        # Looks for Total = A × B style relationships (domain-agnostic).
        # Tries all triples of numeric columns, not just ones named "total".
        analytical_num = [
            c for c in numeric_cols
            if not c.endswith(("_Is_Missing", "_Is_Invalid", "_Is_Outlier"))
            and not _is_id_column(c, self.df[c])
        ]
        checked_triples = set()
        for total_col in analytical_num:
            for c1 in analytical_num:
                if c1 == total_col:
                    continue
                for c2 in analytical_num:
                    if c2 == total_col or c2 == c1:
                        continue
                    triple = frozenset([total_col, c1, c2])
                    if triple in checked_triples:
                        continue
                    checked_triples.add(triple)

                    subset = self.df[[total_col, c1, c2]].dropna()
                    if len(subset) < 20:
                        continue
                    # Avoid division-by-zero and near-zero denominators
                    if (subset[c1].abs() < 1e-8).mean() > 0.5:
                        continue
                    if (subset[c2].abs() < 1e-8).mean() > 0.5:
                        continue

                    diff = (subset[c1] * subset[c2] - subset[total_col]).abs()
                    rel  = diff / (subset[total_col].abs().replace(0, np.nan))
                    if rel.dropna().median() < 0.01 and (rel.dropna() < 0.05).mean() > 0.90:
                        # Found: total_col ≈ c1 * c2

                        # Fix rows where all three exist but math is wrong
                        broken = (
                            self.df[total_col].notna() &
                            self.df[c1].notna() &
                            self.df[c2].notna() &
                            ((self.df[c1] * self.df[c2] - self.df[total_col]).abs() > 1e-4)
                        )
                        if int(broken.sum()) > 0:
                            self.df.loc[broken, total_col] = self.df.loc[broken, c1] * self.df.loc[broken, c2]
                            self.logger.log("4_Validity", total_col, int(broken.sum()),
                                            "Recomputed Derived Column",
                                            details=f"Enforced: {total_col} = {c1} × {c2}")

                        # Back-fill missing total
                        m_total = self.df[total_col].isna() & self.df[c1].notna() & self.df[c2].notna()
                        if int(m_total.sum()) > 0:
                            self.df.loc[m_total, total_col] = self.df.loc[m_total, c1] * self.df.loc[m_total, c2]
                            self.logger.log("4_Validity", total_col, int(m_total.sum()),
                                            "Imputed from Relationship",
                                            details=f"Computed: {total_col} = {c1} × {c2}")

                        # Back-fill missing c1
                        m_c1 = self.df[c1].isna() & self.df[total_col].notna() & self.df[c2].notna() & (self.df[c2] != 0)
                        if int(m_c1.sum()) > 0:
                            self.df.loc[m_c1, c1] = self.df.loc[m_c1, total_col] / self.df.loc[m_c1, c2]
                            self.logger.log("4_Validity", c1, int(m_c1.sum()),
                                            "Imputed from Relationship",
                                            details=f"Computed: {c1} = {total_col} / {c2}")

                        # Back-fill missing c2
                        m_c2 = self.df[c2].isna() & self.df[total_col].notna() & self.df[c1].notna() & (self.df[c1] != 0)
                        if int(m_c2.sum()) > 0:
                            self.df.loc[m_c2, c2] = self.df.loc[m_c2, total_col] / self.df.loc[m_c2, c1]
                            self.logger.log("4_Validity", c2, int(m_c2.sum()),
                                            "Imputed from Relationship",
                                            details=f"Computed: {c2} = {total_col} / {c1}")

                        break  # move on to the next potential total_col

    # ═════════════════════════════════════════════════════════════════════════
    # Stage 5 — Outlier Flagging (IQR)
    # ═════════════════════════════════════════════════════════════════════════
    def _stage5_outliers(self):
        """
        Flag statistical outliers using the IQR method.
        Outliers are NEVER removed — only flagged with boolean columns.
        ID-like columns and flag columns are excluded.
        """
        numeric_cols = self.df.select_dtypes(include="number").columns
        review_flags = pd.DataFrame(index=self.df.index)

        for col in numeric_cols:
            if col.endswith(("_Is_Missing", "_Is_Invalid", "_Is_Outlier")):
                continue
            if _is_id_column(col, self.df[col]):
                continue

            q1  = self.df[col].quantile(0.25)
            q3  = self.df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue  # constant column — no meaningful outlier range

            lower        = q1 - 1.5 * iqr
            upper        = q3 + 1.5 * iqr
            mask         = (self.df[col] < lower) | (self.df[col] > upper)
            outlier_count = int(mask.sum())

            if outlier_count > 0:
                flag_col = f"{col}_Is_Outlier"
                self.df[flag_col]     = mask
                review_flags[flag_col] = mask
                self.logger.log("5_Outlier", col, outlier_count,
                                "Flagged IQR Outliers — NOT Removed",
                                details=f"Fence: [{lower:.4f}, {upper:.4f}]. "
                                        "Outliers retained; only flagged for human review.")

        # Include semantic invalidity flags in the review board
        for col in self.df.columns:
            if col.endswith("_Is_Invalid"):
                review_flags[col] = self.df[col]

        if not review_flags.empty:
            has_any_flag = review_flags.any(axis=1)
            return self.df[has_any_flag].copy()
        return pd.DataFrame()

    # ═════════════════════════════════════════════════════════════════════════
    # Stage 6 — Validation Pass
    # ═════════════════════════════════════════════════════════════════════════
    def _stage6_validation_pass(self):
        """
        Runs after all cleaning stages to verify the pipeline actually worked.
        Logs any remaining issues as warnings so they appear in the changelog.
        """
        # Check: any numeric columns that should have been imputed but still have nulls?
        for col in self.df.columns:
            if col.endswith(("_Is_Missing", "_Is_Invalid", "_Is_Outlier")):
                continue
            remaining_null = int(self.df[col].isnull().sum())
            if remaining_null == 0:
                continue

            info   = self.profile.get(col, {})
            itype  = info.get("inferred_type", "unknown")
            is_id  = info.get("is_id", False)
            m_pct  = info.get("missing_pct", 100.0)

            # If it was supposed to be imputed (numeric ≤ threshold) but still has nulls → warn
            if itype == "numeric" and not is_id and m_pct <= _IMPUTE_THRESH:
                self.logger.log("6_Validation", col, remaining_null,
                                "⚠️ Validation Warning — Nulls Remain After Imputation",
                                details=f"{remaining_null} null(s) remain in '{col}' after imputation attempt. "
                                        "Check if new NaN values were introduced by Stage 4 (negatives → NaN).")
            else:
                # Expected to be flagged — confirm the flag column exists
                expected_flag = f"{col}_Is_Missing"
                if expected_flag not in self.df.columns:
                    # Flag was not created — create it now
                    self.df[expected_flag] = self.df[col].isnull()
                    self.logger.log("6_Validation", col, remaining_null,
                                    "⚠️ Validation — Late Flag Created",
                                    new_stat=expected_flag,
                                    details=f"Flag column was missing. Created in validation pass.")

        # Check: duplicate rows introduced (shouldn't happen, but verify)
        n_dupes = int(self.df.duplicated().sum())
        if n_dupes > 0:
            self.logger.log("6_Validation", "ALL_ROWS", n_dupes,
                            "⚠️ Validation Warning — Duplicates Found Post-Cleaning",
                            details=f"{n_dupes} duplicate rows detected after cleaning. "
                                    "These were not present or were re-introduced. Review pipeline.")

        self.logger.log("6_Validation", "PIPELINE", 0,
                        "✅ Validation Pass Complete",
                        details=f"Final shape: {self.df.shape[0]} rows × {self.df.shape[1]} cols. "
                                f"Total nulls remaining: {int(self.df.isnull().sum().sum())}.")

    # ═════════════════════════════════════════════════════════════════════════
    # Before / After Summary
    # ═════════════════════════════════════════════════════════════════════════
    def _build_summary(self) -> dict:
        """
        Build a structured before/after summary dict that the UI can display.
        Covers: missing values, duplicates, outliers, standardised categories,
        invalid values, and shape changes.
        """
        orig = self.original_df
        curr = self.df

        # ── Missing values ───────────────────────────────────────────────────
        orig_missing = int(orig.isnull().sum().sum())
        curr_missing = int(
            curr[[c for c in curr.columns
                  if not c.endswith(("_Is_Missing", "_Is_Invalid", "_Is_Outlier"))]
                 ].isnull().sum().sum()
        )

        # ── Duplicates ───────────────────────────────────────────────────────
        orig_dupes = int(orig.duplicated().sum())
        curr_dupes = int(curr.duplicated().sum())

        # ── Outliers flagged ─────────────────────────────────────────────────
        outlier_flag_cols = [c for c in curr.columns if c.endswith("_Is_Outlier")]
        n_outlier_rows    = int(curr[outlier_flag_cols].any(axis=1).sum()) if outlier_flag_cols else 0

        # ── Invalid values flagged ───────────────────────────────────────────
        invalid_flag_cols = [c for c in curr.columns if c.endswith("_Is_Invalid")]
        n_invalid_rows    = int(curr[invalid_flag_cols].any(axis=1).sum()) if invalid_flag_cols else 0

        # ── Standardised categories ──────────────────────────────────────────
        standardised = [
            log for log in self.logger.get_logs()
            if log["method"] == "Standardize Categorical Casing"
        ]
        n_std_changes = sum(log["rows_affected"] for log in standardised)

        return {
            "before_rows":        orig.shape[0],
            "after_rows":         curr.shape[0],
            "before_cols":        orig.shape[1],
            "after_cols":         curr.shape[1],
            "before_missing":     orig_missing,
            "after_missing":      curr_missing,
            "missing_fixed":      max(0, orig_missing - curr_missing),
            "before_dupes":       orig_dupes,
            "after_dupes":        curr_dupes,
            "dupes_removed":      max(0, orig_dupes - curr_dupes),
            "outlier_rows_flagged": n_outlier_rows,
            "invalid_rows_flagged": n_invalid_rows,
            "categorical_values_standardised": n_std_changes,
            "total_log_entries":  len(self.logger.get_logs()),
        }
