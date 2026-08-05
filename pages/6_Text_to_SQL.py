"""
pages/6_Text_to_SQL.py
----------------------
Allows users to ask natural language questions about their dataset.
Uses Groq to translate the question to DuckDB SQL, runs the query, and displays results.
"""

import streamlit as st
import pandas as pd
import duckdb
import sys
import os

from groq import Groq

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with Data · Data Analyst",
    page_icon="💬",
    layout="wide",
)

if os.path.dirname(__file__) + "/.." not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__) + "/..")
from utils.ui import inject_custom_css, sidebar_brand
inject_custom_css()
sidebar_brand()

st.title("💬 Chat with your Data (Text-to-SQL)")
st.markdown("Ask questions in plain English. Our AI will write the SQL and execute it instantly against your uploaded dataset.")

if "df" not in st.session_state:
    st.warning("⬆️ Please upload a dataset first on the **Upload Data** page.")
    st.stop()

df = st.session_state["df"]

# Show Schema
with st.expander("🔍 View Dataset Schema (Columns & Types)", expanded=False):
    schema_df = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str)})
    st.dataframe(schema_df, hide_index=True, use_container_width=True)

# ── Setup Groq API ───────────────────────────────────────────────────────────
groq_api_key = None
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    groq_api_key = os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("⚠️ Groq API key not found. Please add `GROQ_API_KEY` to your Streamlit secrets or environment variables.")
    st.stop()

client = Groq(api_key=groq_api_key)

# ── Helper: LLM SQL Generation ───────────────────────────────────────────────
def generate_sql(question: str, columns: list, dtypes: list) -> str:
    schema_str = ", ".join([f"{col} ({dt})" for col, dt in zip(columns, dtypes)])
    
    prompt = f"""You are an expert Data Analyst and SQL developer.
I have a table called `df`.
Here is the schema (column name and type):
{schema_str}

The user wants to know: "{question}"

Write a DuckDB-compatible SQL query to answer this question. 
Return ONLY the raw SQL query. Do not include markdown formatting like ```sql or explanations. Just the query itself.
Example: SELECT category, SUM(revenue) FROM df GROUP BY category ORDER BY SUM(revenue) DESC LIMIT 5;
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You output only raw SQL queries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    query = response.choices[0].message.content.strip()
    # Strip markdown if the LLM accidentally includes it
    if query.startswith("```sql"):
        query = query[6:]
    if query.startswith("```"):
        query = query[3:]
    if query.endswith("```"):
        query = query[:-3]
    return query.strip()

# ── UI: Ask Question ─────────────────────────────────────────────────────────
st.divider()
question = st.text_input("Ask a question about your data:", placeholder="e.g., What is the total revenue by category? Limit to top 5.")

if st.button("Generate & Run SQL", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("AI is writing the SQL query..."):
            try:
                # Generate SQL
                sql_query = generate_sql(question, df.columns.tolist(), df.dtypes.astype(str).tolist())
                
                st.markdown("### Generated SQL Query")
                st.code(sql_query, language="sql")
                
                with st.spinner("Executing query with DuckDB..."):
                    # Execute with DuckDB
                    # DuckDB automatically finds the local variable `df` and queries it
                    result_df = duckdb.query(sql_query).df()
                    
                st.markdown("### Results")
                if result_df.empty:
                    st.info("The query returned no results.")
                else:
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    
                    # Download button
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Results (.csv)",
                        csv,
                        "query_results.csv",
                        "text/csv"
                    )
            except Exception as e:
                st.error("❌ An error occurred while generating or running the query.")
                st.error(str(e))
                st.info("Try rephrasing your question to be more specific about the column names.")
