import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# Configure API Key

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

st.title("🤖 AI Business Insights")

if "df" not in st.session_state:
    st.warning("Upload a dataset first.")
    st.stop()

df = st.session_state["df"]

st.subheader("Dataset Preview")

st.dataframe(df.head())

if st.button("Generate AI Insights"):

    with st.spinner("Analyzing dataset..."):

        summary = df.describe(include='all').to_string()

        prompt = f"""
You are a Senior Data Analyst.

Analyze this dataset summary.

{summary}

Generate:

1. Executive Summary

2. Key Trends

3. Potential Risks

4. Business Opportunities

5. Recommendations

Keep the response business-focused.
"""

        response = model.generate_content(prompt)

        st.success("Analysis Complete!")

        st.markdown(response.text)