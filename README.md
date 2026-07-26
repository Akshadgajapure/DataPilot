# 🤖 AI-Powered Data Analyst Assistant

An AI-powered analytics application that automates **Exploratory Data Analysis (EDA)**, interactive dashboard creation, and **AI-generated business insights** from structured datasets.

Built using **Python, Streamlit, Pandas, Plotly, and Google Gemini AI**, the application enables users to upload CSV or Excel files, analyze business data, visualize KPIs, and generate actionable insights within minutes.

---

# 📸 Project Preview

## 🏠 Home Page

![Home](assets/home.png)

---

## 📂 Upload Dataset

![Upload Dataset](assets/upload.png)

---

## 📊 Exploratory Data Analysis

![EDA](assets/analysis.png)

---

## 📈 Interactive Dashboard

![Dashboard](assets/dashboard.png)

---

## 🤖 AI Business Insights

![AI Insights](assets/ai_insights.png)

---

# ✨ Features

### 📂 Data Upload
- Upload CSV & Excel datasets; data is shared across all pages via session state

### 🧹 Data Cleaning Studio
- **Interactive Cleaning**: Drop duplicates and remove unnecessary columns
- **Handle Missing Values**: Impute missing data with mean, median, mode, or custom values, or drop rows with missing values
- **Apply Globally**: Update the dataset across all other pages (EDA, Dashboard, AI Insights) with one click
- **Export**: Download the cleaned dataset as a fresh CSV file

### 📊 Automated EDA — Data Profiling Report
- **Per-column profile table**: dtype, % missing, unique count, cardinality (low/medium/high)
- Numeric stats: min, max, mean, median, std, skewness, **outlier count (IQR method)**
- Categorical stats: top-5 values with frequency %, **ID-column detection** (near-100% unique)
- Datetime stats: min/max date, span in days, **gap detection**
- **Data Quality Warnings** — auto-generated: >30% missing, constant columns, duplicate rows, case-variant detection (e.g. "Male"/"male"/"M"), highly correlated pairs (|r| > 0.85)
- **Distribution grid** — histograms for every numeric column, bar charts for every categorical, rendered 3-per-row automatically
- **Plain-language Dataset Summary** — computed from real stats, not LLM guessing
- Correlation heatmap with high-correlation pair annotations

### 📈 Fully Interactive Dashboard
- **Persistent filter sidebar**: date range picker, multi-select per categorical column, range slider per numeric column — all AND-combined, updates charts live
- **Reset Filters** button clears all active filters in one click
- **KPI cards** with delta vs unfiltered baseline (period-over-period trend when a date column exists)
- **Switchable chart types**: Bar, Line, Scatter, Box Plot, Correlation Heatmap
- Axis dropdowns update valid options per chart type (e.g. Box Plot Y must be numeric)
- **Trend / Line chart warning**: detects non-datetime X axis and alerts the user
- **Drill-down**: click bars or scatter points → a detail chart + data table appear below
- Large datasets (> 50 000 rows) get an explicit **Apply Filters** button

### 🤖 AI-Powered Business Insights
- Powered by **Groq API** (`llama-3.3-70b-versatile`) for fast, high-quality analysis
- Prompt is grounded in the **computed EDA profile** when EDA page is visited first
- Generates: Executive Summary, Key Trends, Risks, Opportunities, Recommendations

### 🛠️ Code Quality
- Profiling logic in `utils/data_profiler.py` — pure functions, no Streamlit dependency
- Dashboard logic in `utils/dashboard_helpers.py` — pure filter + axis helpers
- Every function has a docstring explaining inputs, outputs, and design decisions

---

# 📊 Sample Dataset

The project is demonstrated using an **E-commerce Transactions Dataset** containing:

- User ID
- Product ID
- Product Category
- Product Price
- Discount (%)
- Final Price
- Payment Method
- Purchase Date

The application can analyze **any structured CSV or Excel dataset**.

---

# 🛠️ Tech Stack

### Language

- Python

### Libraries & Frameworks

- Streamlit
- Pandas
- Plotly
- OpenPyXL
- Groq Python SDK (LLM inference via `llama-3.3-70b-versatile`)

### Concepts

- Exploratory Data Analysis (EDA) & Automated Data Profiling
- Business Intelligence & Dashboard Development
- Data Visualization & Interactive Filtering
- AI-powered Analytics

---

# 📂 Project Structure

```text
AI-Powered-Data-Analyst-Assistant
│
├── app.py                          # Home page
├── README.md
├── requirements.txt
├── .gitignore
│
├── .streamlit/
│   └── secrets.toml                # API keys (gitignored — never committed)
│
├── utils/                          # Shared business logic (no Streamlit code)
│   ├── __init__.py
│   ├── data_profiler.py            # EDA profiling engine (pure functions)
│   └── dashboard_helpers.py        # Filter application + chart axis helpers
│
├── pages/
│   ├── 1_Upload_Data.py            # File upload → st.session_state["df"]
│   ├── 2_Data_Analysis.py          # Automated data profiling report
│   ├── 3_Dashboard.py              # Fully interactive filtered dashboard
│   ├── 4_AI_Insights.py            # Groq-powered business insights
│   └── 5_Data_Cleaning.py          # Interactive dataset cleaning & export
│
├── assets/
│   ├── home.png
│   ├── upload.png
│   ├── analysis.png
│   ├── dashboard.png
│   └── ai_insights.png
│
└── datasets/
    └── ecommerce_dataset_updated.csv
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/ashikaas/AI-Powered-Data-Analyst-Assistant.git
```

Move into the project

```bash
cd AI-Powered-Data-Analyst-Assistant
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

The application will launch at:

```text
http://localhost:8501
```

---

# 🌐 Deployment

The application can be deployed on **Streamlit Community Cloud**.

1. Push the project to GitHub.
2. Connect the repository to Streamlit Community Cloud.
3. Set **app.py** as the entry point.
4. Add your **GOOGLE_API_KEY** under **App Settings → Secrets**.
5. Deploy and share the live application.

---

# 📈 AI-Generated Insights

The application can generate:

- Executive Summary
- Key Trends
- Business Opportunities
- Potential Risks
- Business Recommendations

---

# 🚀 Future Enhancements

- Export AI insights as PDF
- Dashboard report download
- Database connectivity (PostgreSQL/MySQL)
- Multi-file dataset analysis
- Advanced statistical analytics
- Custom AI prompts

---

# 👨‍💻 Author

**Ashika Srivastava**

B.Tech Computer Science Engineering

Interested in **Data Analytics, Product Analytics, Business Intelligence, and AI-powered Analytics**.

---

## ⭐ If you found this project useful, consider giving it a Star!