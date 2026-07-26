"""
pages/4_AI_Insights.py
----------------------
AI-Powered Insights page with two sections:

1. 📊 Auto-generated Business Insights Report (one-click, Groq LLM)
2. 💬 Dataset Chatbot — persistent chat where users ask any question
   about their data; the LLM has full context of the dataset profile
   and can also run basic pandas queries to answer factual questions.
"""

import streamlit as st
import pandas as pd
import io
from datetime import datetime
import os
import sys
from groq import Groq
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS
import base64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.pdf_report import generate_pdf_report

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Insights · Data Analyst",
    page_icon="🤖",
    layout="wide",
)

import sys, os
if os.path.dirname(__file__) + "/.." not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__) + "/..")
from utils.ui import inject_custom_css, sidebar_brand
inject_custom_css()
sidebar_brand()

# ── Groq client ───────────────────────────────────────────────────────────────
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error(
        "⚠️ Groq API key not found. "
        "Add `GROQ_API_KEY = '...'` to `.streamlit/secrets.toml`."
    )
    st.stop()

MODEL = "llama-3.3-70b-versatile"

# ── Guard: dataset must be uploaded ──────────────────────────────────────────
if "df" not in st.session_state:
    st.warning("⬆️ Please upload a dataset first (Upload Data page).")
    st.stop()

# ── Always use the cleaned dataset if available ─────────────────────────────
# This ensures the AI report reflects post-cleaning reality, not the raw messy data.
df: pd.DataFrame = st.session_state.get("clean_df", st.session_state["df"])
data_source_label = "✅ Using **Cleaned Dataset**" if "clean_df" in st.session_state else "⚠️ Using **Raw Dataset** (run Data Cleaning for better insights)"
st.caption(data_source_label)

# ── Build a rich, data-aware context string ───────────────────────────────────
def build_context(df: pd.DataFrame) -> str:
    """
    Compose a data-aware, information-dense context string.
    - Uses actual post-cleaning stats
    - Detects business KPIs (revenue, volume, dates)
    - Uses smarter correlation: filters out mathematically forced pairs
    """
    import numpy as np
    buf = io.StringIO()

    buf.write(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    buf.write(f"Columns: {', '.join(df.columns.tolist())}\n\n")

    # ── Data Types ─────────────────────────────────────────────────────────────
    buf.write("--- Data Types ---\n")
    buf.write(df.dtypes.to_string() + "\n\n")

    # ── Missing Values ─────────────────────────────────────────────────────────
    buf.write("--- Missing Values ---\n")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        buf.write(missing.to_string() + "\n\n")
    else:
        buf.write("No missing values (dataset is complete).\n\n")

    # ── Numeric Summary ────────────────────────────────────────────────────────
    buf.write("--- Numeric Summary (Post-Cleaning) ---\n")
    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        buf.write(num_df.describe().round(2).to_string() + "\n\n")

    # ── Business KPI Detection ─────────────────────────────────────────────────
    buf.write("--- Business KPIs (Auto-Detected) ---\n")
    kpi_keywords = {
        "revenue": ["revenue", "sales", "income", "earnings"],
        "cost":    ["cost", "expense", "spend", "price"],
        "volume":  ["quantity", "qty", "units", "count", "orders"],
        "customer":["customer", "client", "user"],
        "profit":  ["profit", "margin", "gain"],
    }
    found_kpis = False
    for kpi_type, keywords in kpi_keywords.items():
        for col in num_df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in keywords):
                col_data = df[col].dropna()
                buf.write(
                    f"{col} [{kpi_type.upper()}]: "
                    f"Total={col_data.sum():,.2f}, "
                    f"Mean={col_data.mean():,.2f}, "
                    f"Min={col_data.min():,.2f}, "
                    f"Max={col_data.max():,.2f}\n"
                )
                found_kpis = True
    if not found_kpis:
        buf.write("No standard KPI columns detected by name heuristics.\n")
    buf.write("\n")

    # ── Datetime Analysis ──────────────────────────────────────────────────────
    dt_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    # Also try to detect string columns that look like dates
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(10)
        try:
            pd.to_datetime(sample, format="mixed", errors="raise")
            dt_cols.append(col)
        except Exception:
            pass
    
    if dt_cols:
        buf.write("--- Datetime Analysis ---\n")
        for col in dt_cols[:3]:
            try:
                parsed = pd.to_datetime(df[col], format="mixed", errors="coerce").dropna()
                buf.write(
                    f"{col}: Range [{parsed.min().date()} → {parsed.max().date()}], "
                    f"Span={( parsed.max() - parsed.min()).days} days\n"
                )
            except Exception:
                pass
        buf.write("\n")

    # ── Smart Correlation (filters out mathematically forced pairs) ───────────
    buf.write("--- Key Correlations (Meaningful Pairs Only) ---\n")
    if not num_df.empty and num_df.shape[1] > 1:
        corr_matrix = num_df.corr(numeric_only=True).abs()
        # Get upper triangle pairs
        pairs = []
        cols_list = corr_matrix.columns.tolist()
        for i in range(len(cols_list)):
            for j in range(i + 1, len(cols_list)):
                c1, c2 = cols_list[i], cols_list[j]
                r = corr_matrix.loc[c1, c2]
                if pd.isna(r):
                    continue
                # Skip pairs where one column name is contained in the other
                # (likely a derived column, e.g., Total vs Total_Revenue)
                if c1.lower() in c2.lower() or c2.lower() in c1.lower():
                    continue
                # Skip flag columns generated by the cleaner
                if c1.endswith("_Is_Missing") or c2.endswith("_Is_Missing"):
                    continue
                if c1.endswith("_Is_Outlier") or c2.endswith("_Is_Outlier"):
                    continue
                pairs.append((r, c1, c2))
        pairs.sort(reverse=True)
        if pairs:
            for r, c1, c2 in pairs[:6]:
                direction = "positive" if num_df[[c1, c2]].corr().iloc[0, 1] > 0 else "negative"
                buf.write(f"  {c1} ↔ {c2}: r={r:.3f} ({direction})\n")
        else:
            buf.write("  No meaningful non-derived correlations found.\n")
    buf.write("\n")

    # ── Categorical Columns ───────────────────────────────────────────────────
    buf.write("--- Categorical Columns (top values) ---\n")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols[:6]:
        top = df[col].value_counts().head(5).to_dict()
        buf.write(f"{col}: {top}\n")
    buf.write("\n")

    # ── Sample Rows ───────────────────────────────────────────────────────────
    buf.write("--- First 5 Rows ---\n")
    buf.write(df.head().to_string(index=False) + "\n")

    return buf.getvalue()

DATASET_CONTEXT = build_context(df)

SYSTEM_PROMPT = f"""You are an expert Data Analyst assistant integrated into a data analysis application.
The user has uploaded a dataset. Here is the complete context about the dataset:

{DATASET_CONTEXT}

Your job:
- Answer questions about this specific dataset accurately using the context above.
- When giving numbers, reference actual values from the dataset context.
- If asked for analysis, provide clear, business-focused insights.
- If the user asks something the context cannot answer, say so honestly.
- Keep responses concise but complete. Use markdown formatting.
- Use bullet points, tables (markdown), and headers where helpful.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🤖 AI Insights")
st.caption(f"Powered by **Groq** · `{MODEL}` · Dataset: **{df.shape[0]:,} rows × {df.shape[1]} columns**")

tab1, tab2 = st.tabs(["📊 Business Insights Report", "💬 Dataset Chatbot"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Auto-generated Business Insights Report
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📊 Auto-generated Business Insights")
    st.markdown(
        "Click the button to let the AI analyse your dataset and generate "
        "a structured business report."
    )

    col_btn, col_clear = st.columns([2, 1])
    generate = col_btn.button("🚀 Generate Insights Report", type="primary", use_container_width=True)
    if col_clear.button("🗑️ Clear Report", use_container_width=True):
        if "insights_report" in st.session_state:
            del st.session_state["insights_report"]
        st.rerun()

    if generate:
        with st.spinner("Analysing dataset with Groq AI… this may take 10–20 seconds"):
            report_prompt = f"""You are a Senior Data Analyst. You have been given REAL, COMPUTED statistics from an actual dataset. Your job is to write an evidence-based business report — every claim MUST cite a specific number or column name from the data below.

DO NOT use generic statements like "the data shows trends". EVERY insight must reference an actual value.

=== DATASET CONTEXT (POST-CLEANING) ===
{DATASET_CONTEXT}
=== END CONTEXT ===

Generate a well-structured report with these exact sections:

## 1. Executive Summary
In 3–4 sentences, describe what this dataset contains, its time range (if dates exist), total record count, and the single most important business fact you found.

## 2. Business KPIs
List the key numeric KPIs with their ACTUAL values (total, mean, min, max). Cite column names and numbers directly.

## 3. Key Trends & Patterns
What patterns or distributions stand out? Reference actual values, percentages, or date ranges.

## 4. Correlation Insights
Explain the meaningful correlations found. What do they imply for the business? Skip any obvious mathematical derivations.

## 5. Data Quality Assessment
Comment on missing values, outliers, or anomalies found. Be specific about percentages and affected columns.

## 6. Business Opportunities
Based on the ACTUAL data findings above, what are 3 specific, actionable opportunities?

## 7. Risks & Concerns
What data-backed risks should the business be aware of?

## 8. Recommendations
Provide 3–5 concrete, prioritised recommendations with evidence from the data.

IMPORTANT: Every section must cite specific numbers, column names, or row counts from the context above. No generic template language."""

            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": report_prompt}],
                    temperature=0.3,
                    max_tokens=2048,
                )
                st.session_state["insights_report"] = resp.choices[0].message.content
            except Exception as e:
                st.error(f"Groq API error: {e}")

    if "insights_report" in st.session_state:
        st.divider()
        st.markdown(st.session_state["insights_report"])

        st.divider()
        st.markdown("### 📄 Download Report")
        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            # Plain text fallback
            st.download_button(
                label="⬇️ Download as .txt",
                data=st.session_state["insights_report"],
                file_name="ai_insights_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with dl_col2:
            # Full industry-grade PDF
            if st.button("📄 Generate & Download PDF Report", type="primary", use_container_width=True):
                with st.spinner("Building industry-grade PDF report…"):
                    try:
                        dataset_name = st.session_state.get("dataset_filename", "Dataset")
                        cleaning_fixes = st.session_state.get("ai_clean_fixes", [])
                        pdf_bytes = generate_pdf_report(
                            df=df,
                            dataset_name=dataset_name,
                            ai_insights=st.session_state["insights_report"],
                            cleaning_fixes=cleaning_fixes if cleaning_fixes else None,
                        )
                        st.session_state["pdf_report_bytes"] = pdf_bytes
                        st.success("✅ PDF generated successfully!")
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
                        import traceback; st.code(traceback.format_exc())

        if "pdf_report_bytes" in st.session_state:
            fname = f"data_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button(
                label="⬇️ Click Here to Download PDF",
                data=st.session_state["pdf_report_bytes"],
                file_name=fname,
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Dataset Chatbot
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 💬 Chat with your Dataset")
    st.markdown(
        "Ask anything about your data in plain English. "
        "The AI has full context of every column, statistics, and sample rows."
    )

    # Initialise chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Suggested starter questions
    with st.expander("💡 Example questions you can ask", expanded=False):
        st.markdown("""
- *What are the most common categories in this dataset?*
- *Which numeric column has the highest average value?*
- *Are there any missing values I should worry about?*
- *What does this dataset seem to be about?*
- *Which column would be the best target variable for ML?*
- *Summarise the top trends in this data.*
- *What are the outliers in the Price column?*
- *How many unique customers are in this dataset?*
        """)

    # Clear chat button
    if st.button("🗑️ Clear Chat History", key="clear_chat"):
        st.session_state["chat_history"] = []
        st.rerun()

    st.divider()

    # ── Render existing messages ──────────────────────────────────────────────
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask anything about your dataset…")

    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("**Click to speak:**")
        audio_bytes = audio_recorder(icon_size="2x", icon_name="microphone-lines")

    if audio_bytes and ("last_audio" not in st.session_state or st.session_state["last_audio"] != audio_bytes):
        st.session_state["last_audio"] = audio_bytes
        with st.spinner("🎙️ Transcribing audio..."):
            try:
                transcription = client.audio.transcriptions.create(
                    file=("audio.wav", audio_bytes),
                    model="whisper-large-v3",
                    prompt="The user is asking a data analysis question."
                )
                user_input = transcription.text
            except Exception as e:
                st.error(f"Transcription failed: {e}")

    if user_input:
        # Show user message immediately
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        # Build messages for Groq (system + full history)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in st.session_state["chat_history"]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Stream the response
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1024,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    full_response += delta
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"⚠️ Groq API error: {e}"
                response_placeholder.error(full_response)

        # Save assistant reply to history
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": full_response}
        )

        # Generate Text-to-Speech (TTS)
        try:
            tts = gTTS(text=full_response, lang='en')
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            b64 = base64.b64encode(fp.read()).decode()
            md = f"""
                <audio autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(md, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Audio playback failed: {e}")