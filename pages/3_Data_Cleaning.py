"""
pages/5_Data_Cleaning.py
------------------------
Strict, Defensible Data Cleaning Pipeline

Architecture: Rule-based engine that adheres strictly to:
- Never silently fabricate data
- Never break internal consistency
- Every transformation is logged
- Flag before fixing
"""

import streamlit as st
import pandas as pd
import uuid
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_cleaning_engine import StrictDataCleaner

st.set_page_config(
    page_title="Data Cleaning · Data Analyst",
    page_icon="🧹",
    layout="wide",
)

import sys, os
if os.path.dirname(__file__) + "/.." not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__) + "/..")
from utils.ui import inject_custom_css, sidebar_brand
inject_custom_css()
sidebar_brand()

st.title("🧹 Strict Data Cleaning Pipeline")
st.markdown("A highly defensible, industry-grade pipeline that cleans your data without silent corruption or hallucination. **Every change is logged.**")

if "df" not in st.session_state:
    st.warning("⬆️ Please upload a dataset first on the **Upload Data** page.")
    st.stop()

# Initialize a working copy of the dataframe if not present
if "clean_df" not in st.session_state or st.session_state.get("dataset_id") != st.session_state.get("clean_dataset_id"):
    st.session_state["clean_df"] = st.session_state["df"].copy()
    st.session_state["clean_dataset_id"] = st.session_state.get("dataset_id")
    # Reset state variables on new data
    for key in ["changelog_df", "outlier_df", "cleaning_profile"]:
        st.session_state.pop(key, None)

df: pd.DataFrame = st.session_state["clean_df"]

# ─── Top Metrics ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Columns", f"{df.shape[1]:,}")
c3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
c4.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

st.divider()

# ─── Pipeline Execution ──────────────────────────────────────────────────────
st.subheader("⚙️ Run Strict Pipeline")
st.markdown("""
This pipeline will execute 5 stages:
1. **Profiling:** Detect types and mathematical relationships.
2. **Structural:** Standardize names, strip whitespace, parse dates.
3. **Missing Values:** Defensible imputation (<5%) or explicit flagging (>=5%).
4. **Validity:** Check domain heuristics (no negative prices) and enforce relationships.
5. **Outliers:** Flag statistical outliers for manual review (IQR method).
""")

if st.button("🚀 Execute Strict Cleaning Pipeline", type="primary", use_container_width=True):
    with st.spinner("Executing strict rules and logging changes..."):
        try:
            cleaner = StrictDataCleaner(df.copy())
            cleaned_df, changelog_df, outlier_df, profile = cleaner.run_pipeline()
            
            st.session_state["clean_df"] = cleaned_df
            st.session_state["changelog_df"] = changelog_df
            st.session_state["outlier_df"] = outlier_df
            st.session_state["cleaning_profile"] = profile
            
            st.success("✅ Pipeline execution complete!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")
            import traceback; st.code(traceback.format_exc())

# ─── Reports ─────────────────────────────────────────────────────────────────
if "changelog_df" in st.session_state:
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["📝 Change Log", "⚠️ Outliers (Flagged)", "📊 Final Data Preview"])
    
    changelog_df = st.session_state["changelog_df"]
    outlier_df = st.session_state["outlier_df"]
    
    with tab1:
        st.markdown("### Machine-Readable Change Log")
        st.markdown("Every transformation made to the dataset is documented below.")
        if changelog_df.empty:
            st.info("No changes were necessary. The dataset was already clean based on strict rules.")
        else:
            st.dataframe(changelog_df, use_container_width=True, hide_index=True)
            
            # Download JSON
            json_log = changelog_df.to_json(orient="records", indent=2)
            st.download_button("⬇️ Download Change Log (.json)", json_log, file_name="cleaning_changelog.json", mime="application/json")
            
    with tab2:
        st.markdown("### Human Review Required: Outliers")
        st.markdown("These rows contain statistically anomalous values. They have **NOT** been removed or capped. We have flagged them with boolean columns so you can review them defensively.")
        if outlier_df.empty:
            st.success("No statistical outliers detected.")
        else:
            st.warning(f"{len(outlier_df)} rows contain at least one outlier flag.")
            st.dataframe(outlier_df, use_container_width=True)
            st.download_button("⬇️ Download Flagged Rows (.csv)", outlier_df.to_csv(index=False), file_name="flagged_outliers.csv", mime="text/csv")
            
    with tab3:
        st.markdown("### Cleaned Data Preview")
        st.dataframe(st.session_state["clean_df"].head(100), use_container_width=True)

st.divider()

# ─── Actions & Download ──────────────────────────────────────────────────────
st.subheader("💾 Save & Deploy")

left, right = st.columns(2)

with left:
    st.markdown("### Apply to App")
    st.markdown("Push this strictly cleaned dataset to the Analytics and Dashboard pages.")
    if st.button("🔄 Apply Changes to App globally", type="primary", use_container_width=True):
        st.session_state["df"] = st.session_state["clean_df"].copy()
        st.session_state["dataset_id"] = str(uuid.uuid4())
        for key in ["dataset_profile_summary", "dataset_profile"]:
            st.session_state.pop(key, None)
        st.success("✅ App updated with the clean dataset!")

with right:
    st.markdown("### Download Cleaned Data")
    st.markdown("Export the final dataset to your computer.")
    csv_data = st.session_state["clean_df"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Cleaned Dataset (.csv)",
        data=csv_data,
        file_name="strict_cleaned_dataset.csv",
        mime="text/csv",
        use_container_width=True,
    )
