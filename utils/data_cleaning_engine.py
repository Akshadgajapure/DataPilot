"""
utils/data_cleaning_engine.py
-----------------------------
Industry-Grade Strict Data Cleaning Pipeline.

Principles:
1. Never silently fabricate or corrupt data.
2. Missing != always fixable.
3. Never break internal consistency.
4. Every transformation must be logged.
5. Flag before you fix, when uncertain.
6. Validate business/domain logic.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import re

class ChangeLogger:
    def __init__(self):
        self.logs = []
        
    def log(self, stage, column, rows_affected, method, old_stat=None, new_stat=None, details=""):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "column": column,
            "rows_affected": int(rows_affected),
            "method": method,
            "old_stat": str(old_stat) if old_stat is not None else None,
            "new_stat": str(new_stat) if new_stat is not None else None,
            "details": details
        }
        self.logs.append(entry)
        
    def get_logs(self):
        return self.logs
        
    def to_dataframe(self):
        return pd.DataFrame(self.logs)

class StrictDataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.original_df = df.copy()
        self.df = df.copy()
        self.logger = ChangeLogger()
        self.profile = {}
        
    def run_pipeline(self):
        """Runs the strict 5-stage cleaning pipeline."""
        self._stage1_profile()
        self._stage2_structural()
        self._stage3_missing()
        self._stage4_validity()
        outlier_df = self._stage5_outliers()
        
        return self.df, self.logger.to_dataframe(), outlier_df, self.profile

    # ── Stage 1: Profiling ───────────────────────────────────────────────────
    def _stage1_profile(self):
        """Detect column types, missing %, and relationships before touching anything."""
        for col in self.df.columns:
            series = self.df[col]
            missing_pct = series.isnull().mean() * 100
            
            # Simple heuristic type detection
            inferred_type = "object"
            if pd.api.types.is_numeric_dtype(series):
                inferred_type = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(series):
                inferred_type = "datetime"
            else:
                if series.nunique() < 20 and len(series) > 100:
                    inferred_type = "categorical"
                else:
                    # check if looks like date
                    sample = series.dropna().head(20).astype(str)
                    if len(sample) > 0:
                        try:
                            pd.to_datetime(sample, format='mixed', errors='raise')
                            inferred_type = "datetime_string"
                        except Exception:
                            pass
                            
            self.profile[col] = {
                "inferred_type": inferred_type,
                "missing_pct": missing_pct,
                "nunique": series.nunique()
            }

    # ── Stage 2: Structural Cleaning ──────────────────────────────────────────
    def _stage2_structural(self):
        """Standardize column names, remove dupes, fix obvious types."""
        # Standardize column names (strip whitespace)
        old_cols = list(self.df.columns)
        self.df.columns = self.df.columns.str.strip()
        for old_c, new_c in zip(old_cols, self.df.columns):
            if old_c != new_c:
                self.logger.log("2_Structural", old_c, len(self.df), "Rename Column", old_c, new_c, "Stripped whitespace")

        # Remove exact duplicates
        before_len = len(self.df)
        self.df.drop_duplicates(inplace=True)
        after_len = len(self.df)
        if before_len > after_len:
            self.logger.log("2_Structural", "ALL", before_len - after_len, "Drop Exact Duplicates", before_len, after_len)

        # Fix types based on profile
        for col, info in self.profile.items():
            col = col.strip() # use new name
            series = self.df[col]
            
            # Strip string whitespace
            if series.dtype == 'object':
                before_strip = series.copy()
                self.df[col] = series.str.strip()
                changed = (before_strip != self.df[col]).sum()
                if changed > 0:
                    self.logger.log("2_Structural", col, changed, "Strip Whitespace", details="Stripped leading/trailing whitespace")

            # Parse datetimes
            if info["inferred_type"] == "datetime_string":
                try:
                    self.df[col] = pd.to_datetime(self.df[col], format='mixed', errors='coerce')
                    self.logger.log("2_Structural", col, len(self.df), "Parse Datetime")
                except Exception:
                    pass

            # Standardize casing for categoricals
            if info["inferred_type"] == "categorical" and self.df[col].dtype == 'object':
                # Convert to title case for consistency if it's string
                try:
                    old_vals = self.df[col].copy()
                    self.df[col] = self.df[col].str.title()
                    changed = (old_vals != self.df[col]).sum()
                    if changed > 0:
                        self.logger.log("2_Structural", col, changed, "Standardize Categorical Case", details="Converted to Title Case")
                except Exception:
                    pass

    # ── Stage 3: Missing Value Handling ───────────────────────────────────────
    def _stage3_missing(self):
        """Defensible missing value handling based on thresholds."""
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count == 0:
                continue
                
            missing_pct = (missing_count / len(self.df)) * 100
            
            # Principle: Under 5% -> statistical imputation. Over 5% -> Flag and leave null.
            if missing_pct <= 5.0:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    # Check skewness to decide mean vs median
                    skew = self.df[col].skew()
                    if pd.notna(skew) and abs(skew) > 1:
                        val = self.df[col].median()
                        method = "Median Imputation (Highly Skewed)"
                    else:
                        val = self.df[col].mean()
                        method = "Mean Imputation (Normal Dist)"
                        
                    if pd.notna(val):
                        self.df[col] = self.df[col].fillna(val)
                        self.logger.log("3_Missing", col, missing_count, method, new_stat=val, details=f"Missing {missing_pct:.2f}% (<=5%)")
                
                elif self.profile.get(col, {}).get("inferred_type") == "categorical":
                    mode_s = self.df[col].mode()
                    if not mode_s.empty:
                        val = mode_s.iloc[0]
                        self.df[col] = self.df[col].fillna(val)
                        self.logger.log("3_Missing", col, missing_count, "Mode Imputation", new_stat=val, details=f"Missing {missing_pct:.2f}% (<=5%)")
            else:
                # Missing > 5%: Add flag, DO NOT impute blindly.
                flag_col = f"{col}_Is_Missing"
                self.df[flag_col] = self.df[col].isnull()
                self.logger.log("3_Missing", col, missing_count, "Created Missing Flag", new_stat=flag_col, details=f"Missing {missing_pct:.2f}% (>5%). Original left null.")

    # ── Stage 4: Validity & Consistency Checks ────────────────────────────────
    def _stage4_validity(self):
        """Range checks, cross-column math/date logic, and semantic string validation."""
        numeric_cols = self.df.select_dtypes(include="number").columns
        string_cols = self.df.select_dtypes(include="object").columns
        
        # 1. Semantic String Validation (Phone, Email, etc.)
        for col in string_cols:
            col_lower = col.lower()
            
            # Phone Validation
            if 'phone' in col_lower or 'mobile' in col_lower or 'contact' in col_lower:
                # Keep only strings that have enough digits (e.g., at least 7 digits)
                # Flag anything that is clearly not a phone number (e.g., text, garbage)
                not_null_mask = self.df[col].notna()
                if not_null_mask.sum() > 0:
                    # Count digits in the string
                    digit_counts = self.df.loc[not_null_mask, col].astype(str).str.count(r'\d')
                    # A valid phone typically has 7 to 15 digits
                    invalid_mask = (digit_counts < 7) | (digit_counts > 15)
                    
                    invalid_count = invalid_mask.sum()
                    if invalid_count > 0:
                        # Convert pandas Series to boolean mask aligned with original df
                        full_invalid_mask = pd.Series(False, index=self.df.index)
                        full_invalid_mask.loc[not_null_mask] = invalid_mask
                        
                        flag_col = f"{col}_Is_Invalid"
                        self.df[flag_col] = full_invalid_mask
                        # We don't overwrite the original data, we just flag it
                        self.logger.log("4_Validity", col, invalid_count, "Flagged Invalid Phone", new_stat=flag_col, details="Values do not contain 7-15 digits")

            # Email Validation
            elif 'email' in col_lower:
                not_null_mask = self.df[col].notna()
                if not_null_mask.sum() > 0:
                    # Basic email regex check
                    valid_email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
                    is_valid = self.df.loc[not_null_mask, col].astype(str).str.match(valid_email_regex)
                    invalid_mask = ~is_valid
                    
                    invalid_count = invalid_mask.sum()
                    if invalid_count > 0:
                        full_invalid_mask = pd.Series(False, index=self.df.index)
                        full_invalid_mask.loc[not_null_mask] = invalid_mask
                        
                        flag_col = f"{col}_Is_Invalid"
                        self.df[flag_col] = full_invalid_mask
                        self.logger.log("4_Validity", col, invalid_count, "Flagged Invalid Email", new_stat=flag_col, details="Values do not match email regex")
        
        # 2. Heuristic non-negative checks
        NON_NEG_KEYWORDS = ['price', 'revenue', 'cost', 'qty', 'quantity', 'age', 'salary', 'amount', 'distance', 'weight']
        for col in numeric_cols:
            col_lower = col.lower()
            if any(kw in col_lower for kw in NON_NEG_KEYWORDS):
                neg_mask = self.df[col] < 0
                neg_count = neg_mask.sum()
                if neg_count > 0:
                    # Flag them as impossible, replace with NaN (don't blindly floor to 0)
                    self.df.loc[neg_mask, col] = np.nan
                    self.logger.log("4_Validity", col, neg_count, "Invalid Negative Replaced with NaN", details="Violated non-negative domain heuristic")

        # Heuristic relationship checks (e.g. Total = Price * Qty)
        # We look for simple multiplication pairs
        for total_col in numeric_cols:
            if 'total' in total_col.lower() or 'revenue' in total_col.lower():
                # try to find components
                for c1 in numeric_cols:
                    if c1 == total_col: continue
                    for c2 in numeric_cols:
                        if c2 == total_col or c2 == c1: continue
                        
                        # Test if c1 * c2 ~ total_col
                        # We test on non-null subset
                        subset = self.df[[total_col, c1, c2]].dropna()
                        if len(subset) > 10:
                            diff = (subset[c1] * subset[c2] - subset[total_col]).abs()
                            if (diff < 1e-4).mean() > 0.90: # 90% match means likely a relationship
                                # We found a relationship. Recompute to fix broken rows.
                                broken = self.df[total_col].notna() & self.df[c1].notna() & self.df[c2].notna() & (abs(self.df[c1] * self.df[c2] - self.df[total_col]) >= 1e-4)
                                broken_count = broken.sum()
                                if broken_count > 0:
                                    self.df.loc[broken, total_col] = self.df.loc[broken, c1] * self.df.loc[broken, c2]
                                    self.logger.log("4_Validity", total_col, broken_count, "Recomputed Derived Column", details=f"Enforced {total_col} = {c1} * {c2}")
                                break # move to next total_col

    # ── Stage 5: Outlier Handling ─────────────────────────────────────────────
    def _stage5_outliers(self):
        """Flag outliers using IQR, do NOT remove. Also gather semantic invalid flags."""
        numeric_cols = self.df.select_dtypes(include="number").columns
        review_flags = pd.DataFrame(index=self.df.index)
        
        # 1. Statistical Outliers (IQR)
        for col in numeric_cols:
            if 'id' in col.lower() or col.endswith('_Is_Missing') or col.endswith('_Is_Invalid'):
                continue
                
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            
            mask = (self.df[col] < lower) | (self.df[col] > upper)
            outlier_count = mask.sum()
            
            if outlier_count > 0:
                flag_col = f"{col}_Is_Outlier"
                self.df[flag_col] = mask
                review_flags[flag_col] = mask
                self.logger.log("5_Outlier", col, outlier_count, "Flagged IQR Outliers", details=f"Range: [{lower:.2f}, {upper:.2f}]")

        # 2. Add existing semantic invalid flags to the review board
        for col in self.df.columns:
            if col.endswith('_Is_Invalid'):
                review_flags[col] = self.df[col]

        # Return a dataframe containing ONLY the rows that have at least one outlier/invalid flag
        if not review_flags.empty:
            has_any_flag = review_flags.any(axis=1)
            return self.df[has_any_flag].copy()
        return pd.DataFrame()
