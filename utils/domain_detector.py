"""
utils/domain_detector.py
------------------------
Dataset domain detection and domain-aware KPI classification.

Detects the likely domain of a dataset (HR, Finance, E-commerce, Healthcare,
Marketing, Education, Manufacturing, or General) from column names, then
provides domain-appropriate KPI labels and metric interpretations.

This module is intentionally dataset-agnostic — it uses keyword heuristics
to classify, but always provides sensible fallbacks for unknown domains.
"""

import io
import pandas as pd
import numpy as np


# ── Domain keyword signatures ─────────────────────────────────────────────────
_DOMAIN_SIGNATURES = {
    "hr": {
        "keywords": [
            "employee", "attrition", "department", "job_role", "job_level",
            "years_at_company", "monthly_income", "work_life", "job_satisfaction",
            "performance_rating", "overtime", "tenure", "compensation",
            "hire", "termination", "workforce", "headcount", "payroll",
            "staff", "personnel", "leave", "absence", "seniority",
            "daily_rate", "hourly_rate", "monthly_rate",
        ],
        "label": "Human Resources / People Analytics",
        "icon": "👥",
    },
    "finance": {
        "keywords": [
            "loan", "interest_rate", "account", "credit", "debit",
            "deposit", "withdrawal", "transaction", "mortgage",
            "investment", "portfolio", "dividend", "stock", "bond",
            "equity", "fund", "asset", "liability", "apr", "emi",
            "principal", "collateral", "default", "risk_score",
        ],
        "label": "Finance / Banking",
        "icon": "🏦",
    },
    "ecommerce": {
        "keywords": [
            "order", "product", "price", "revenue", "cart", "shipping",
            "delivery", "sku", "discount", "coupon", "refund",
            "payment", "checkout", "purchase", "sales",
            "catalog", "inventory", "warehouse", "fulfillment",
        ],
        "label": "E-commerce / Retail",
        "icon": "🛒",
    },
    "healthcare": {
        "keywords": [
            "patient", "diagnosis", "treatment", "hospital", "medication",
            "prescription", "blood_pressure", "heart_rate", "bmi",
            "cholesterol", "symptom", "disease", "clinical", "medical",
            "health", "physician", "surgery", "admission", "discharge",
        ],
        "label": "Healthcare / Medical",
        "icon": "🏥",
    },
    "marketing": {
        "keywords": [
            "campaign", "impression", "click", "conversion", "ctr", "cpc",
            "cpm", "bounce_rate", "engagement", "lead", "channel",
            "ad_spend", "audience", "subscriber", "reach", "funnel",
        ],
        "label": "Marketing / Advertising",
        "icon": "📣",
    },
    "education": {
        "keywords": [
            "student", "grade", "gpa", "course", "enrollment", "teacher",
            "school", "exam", "semester", "major", "degree",
            "attendance", "tuition", "scholarship", "faculty",
        ],
        "label": "Education",
        "icon": "🎓",
    },
    "manufacturing": {
        "keywords": [
            "defect", "production", "machine", "yield", "downtime",
            "maintenance", "batch", "inspection", "tolerance",
            "assembly", "supply_chain", "factory", "oee",
            "cycle_time", "throughput", "scrap",
        ],
        "label": "Manufacturing / Operations",
        "icon": "🏭",
    },
}

# ── KPI semantic mapping per domain ──────────────────────────────────────────
# Maps column name keywords to domain-appropriate KPI types with proper labels.
# "agg" indicates the most meaningful aggregation: "mean", "sum", "rate"
_KPI_SEMANTICS = {
    "hr": [
        {"keywords": ["monthly_income", "salary", "pay", "wage", "compensation",
                       "earnings", "daily_rate", "hourly_rate", "monthly_rate"],
         "label": "Compensation", "icon": "💰", "agg": "mean", "fmt": ",.2f"},
        {"keywords": ["years_at_company", "tenure", "total_working_years",
                       "years_in_role", "years_with_manager", "experience",
                       "years_since_last_promotion", "service_years"],
         "label": "Tenure", "icon": "📅", "agg": "mean", "fmt": ",.1f"},
        {"keywords": ["satisfaction", "job_satisfaction", "environment_satisfaction",
                       "relationship_satisfaction", "work_life_balance"],
         "label": "Satisfaction", "icon": "😊", "agg": "mean", "fmt": ",.2f"},
        {"keywords": ["performance_rating", "performance", "evaluation"],
         "label": "Performance", "icon": "⭐", "agg": "mean", "fmt": ",.2f"},
        {"keywords": ["attrition", "turnover", "churn", "left"],
         "label": "Attrition", "icon": "🚪", "agg": "rate", "fmt": ".1%"},
        {"keywords": ["training_times", "training", "development"],
         "label": "Training", "icon": "📚", "agg": "mean", "fmt": ",.1f"},
        {"keywords": ["distance_from_home", "commute"],
         "label": "Commute Distance", "icon": "🚗", "agg": "mean", "fmt": ",.1f"},
        {"keywords": ["num_companies_worked", "number_of_companies"],
         "label": "Prior Companies", "icon": "🏢", "agg": "mean", "fmt": ",.1f"},
    ],
    "finance": [
        {"keywords": ["loan_amount", "amount", "principal", "balance"],
         "label": "Financial Volume", "icon": "💰", "agg": "sum", "fmt": ",.2f"},
        {"keywords": ["interest_rate", "rate", "apr"],
         "label": "Interest Rate", "icon": "📈", "agg": "mean", "fmt": ",.2f"},
        {"keywords": ["risk_score", "credit_score", "score"],
         "label": "Risk / Credit Score", "icon": "⚠️", "agg": "mean", "fmt": ",.1f"},
        {"keywords": ["income", "annual_income"],
         "label": "Income", "icon": "💵", "agg": "mean", "fmt": ",.2f"},
    ],
    "ecommerce": [
        {"keywords": ["revenue", "sales", "total_price", "total_amount",
                       "earnings", "total_revenue"],
         "label": "Revenue", "icon": "💰", "agg": "sum", "fmt": ",.2f"},
        {"keywords": ["unit_price", "price", "cost"],
         "label": "Price", "icon": "💵", "agg": "mean", "fmt": ",.2f"},
        {"keywords": ["quantity", "qty", "units", "volume"],
         "label": "Volume", "icon": "📦", "agg": "sum", "fmt": ",.0f"},
        {"keywords": ["profit", "margin", "gain"],
         "label": "Profit", "icon": "📈", "agg": "sum", "fmt": ",.2f"},
        {"keywords": ["discount", "discount_pct"],
         "label": "Discount", "icon": "🏷️", "agg": "mean", "fmt": ",.2f"},
        {"keywords": ["satisfaction", "rating", "score"],
         "label": "Customer Satisfaction", "icon": "⭐", "agg": "mean", "fmt": ",.2f"},
    ],
    "healthcare": [
        {"keywords": ["blood_pressure", "systolic", "diastolic"],
         "label": "Blood Pressure", "icon": "❤️", "agg": "mean", "fmt": ",.1f"},
        {"keywords": ["heart_rate", "pulse"],
         "label": "Heart Rate", "icon": "💓", "agg": "mean", "fmt": ",.1f"},
        {"keywords": ["bmi", "body_mass"],
         "label": "BMI", "icon": "⚖️", "agg": "mean", "fmt": ",.1f"},
        {"keywords": ["cholesterol"],
         "label": "Cholesterol", "icon": "🩸", "agg": "mean", "fmt": ",.1f"},
        {"keywords": ["cost", "charge", "bill", "expense"],
         "label": "Healthcare Cost", "icon": "💰", "agg": "mean", "fmt": ",.2f"},
    ],
    "marketing": [
        {"keywords": ["spend", "ad_spend", "budget", "cost"],
         "label": "Ad Spend", "icon": "💰", "agg": "sum", "fmt": ",.2f"},
        {"keywords": ["impression", "reach", "views"],
         "label": "Reach", "icon": "👁️", "agg": "sum", "fmt": ",.0f"},
        {"keywords": ["click", "visits"],
         "label": "Clicks", "icon": "🖱️", "agg": "sum", "fmt": ",.0f"},
        {"keywords": ["conversion", "ctr", "rate"],
         "label": "Conversion Rate", "icon": "🎯", "agg": "mean", "fmt": ".2%"},
    ],
    "education": [
        {"keywords": ["gpa", "grade", "score", "marks"],
         "label": "Academic Performance", "icon": "📊", "agg": "mean", "fmt": ",.2f"},
        {"keywords": ["attendance"],
         "label": "Attendance", "icon": "✅", "agg": "mean", "fmt": ",.1f"},
        {"keywords": ["tuition", "fee", "cost"],
         "label": "Tuition / Cost", "icon": "💰", "agg": "mean", "fmt": ",.2f"},
    ],
    "manufacturing": [
        {"keywords": ["defect", "defect_rate", "scrap"],
         "label": "Defect Rate", "icon": "❌", "agg": "mean", "fmt": ",.2f"},
        {"keywords": ["yield", "oee", "efficiency"],
         "label": "Yield / Efficiency", "icon": "⚙️", "agg": "mean", "fmt": ",.2f"},
        {"keywords": ["throughput", "output", "production"],
         "label": "Throughput", "icon": "🏭", "agg": "sum", "fmt": ",.0f"},
        {"keywords": ["downtime", "cycle_time"],
         "label": "Time Metric", "icon": "⏱️", "agg": "mean", "fmt": ",.2f"},
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def detect_domain(df: pd.DataFrame) -> dict:
    """
    Detect the most likely domain of a dataset from its column names.

    Returns dict with keys: domain, label, icon, confidence, matched_keywords.
    Requires at least 2 keyword matches for a confident classification.
    """
    cols_joined = " ".join(c.lower().replace(" ", "_") for c in df.columns)

    scores = {}
    matches = {}
    for domain, info in _DOMAIN_SIGNATURES.items():
        matched = [kw for kw in info["keywords"] if kw in cols_joined]
        scores[domain] = len(matched)
        matches[domain] = matched

    best = max(scores, key=scores.get) if scores else None
    if best is None or scores[best] < 2:
        return {
            "domain": "general",
            "label": "General Dataset",
            "icon": "📊",
            "confidence": 0.0,
            "matched_keywords": [],
        }

    return {
        "domain": best,
        "label": _DOMAIN_SIGNATURES[best]["label"],
        "icon": _DOMAIN_SIGNATURES[best]["icon"],
        "confidence": round(min(scores[best] / 5, 1.0), 2),
        "matched_keywords": matches[best],
    }


def _is_id_like(col: str, series: pd.Series) -> bool:
    """
    Check if a column looks like an ID/key column.

    Uses both name-based heuristics (suffix/prefix matching) and
    cardinality-based heuristics (>90% unique values).
    """
    col_lower = col.lower().replace(" ", "_")
    if col_lower in ("id", "uuid", "guid"):
        return True

    id_suffixes = ["_id", "_uuid", "_guid", "_key", "_code", "_ref",
                   "_serial", "_token", "_barcode", "_sku"]
    for suffix in id_suffixes:
        if col_lower.endswith(suffix):
            return True

    id_prefixes = ["id_", "uuid_", "guid_"]
    for prefix in id_prefixes:
        if col_lower.startswith(prefix):
            return True

    # Cardinality check: almost every value is unique → likely an ID
    n = len(series.dropna())
    if n > 20 and series.nunique() / n > 0.9:
        return True

    return False


def _compute_stats(data: pd.Series) -> dict:
    """Compute standard descriptive statistics for a numeric series."""
    return {
        "mean":   round(float(data.mean()), 2),
        "median": round(float(data.median()), 2),
        "min":    round(float(data.min()), 2),
        "max":    round(float(data.max()), 2),
        "sum":    round(float(data.sum()), 2),
        "std":    round(float(data.std()), 2) if len(data) > 1 else 0.0,
    }


def classify_kpis(df: pd.DataFrame, domain_info: dict) -> list:
    """
    Classify numeric columns into domain-appropriate KPI categories.

    Phase 1: Match columns against domain-specific KPI templates.
    Phase 2: Add remaining numeric columns as generic KPIs (skip IDs).

    Returns list of KPI dicts with: column, label, icon, agg, fmt, stats.
    """
    domain = domain_info["domain"]
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if not c.endswith(("_Is_Missing", "_Is_Outlier", "_Is_Invalid"))]

    kpis = []
    matched = set()

    # Phase 1: Match columns to domain-specific KPI templates
    templates = _KPI_SEMANTICS.get(domain, [])
    for tpl in templates:
        for col in num_cols:
            if col in matched:
                continue
            cl = col.lower().replace(" ", "_")
            if any(kw in cl for kw in tpl["keywords"]):
                data = df[col].dropna()
                if len(data) == 0:
                    continue

                if tpl["agg"] == "rate":
                    unique_vals = data.unique()
                    if len(unique_vals) <= 3:
                        rate_val = float(data.mean())
                        stats = {
                            "rate": rate_val,
                            "count_positive": int(data.sum()),
                            "count_total": len(data),
                        }
                    else:
                        stats = _compute_stats(data)
                else:
                    stats = _compute_stats(data)

                kpis.append({
                    "column": col, "label": tpl["label"], "icon": tpl["icon"],
                    "agg": tpl["agg"], "fmt": tpl["fmt"], "stats": stats,
                })
                matched.add(col)

    # Phase 2: Add remaining numeric columns as generic KPIs (skip IDs)
    for col in num_cols:
        if col in matched:
            continue
        if _is_id_like(col, df[col]):
            continue
        data = df[col].dropna()
        if len(data) == 0:
            continue

        kpis.append({
            "column": col,
            "label": col.replace("_", " ").title(),
            "icon": "📊",
            "agg": "mean",
            "fmt": ",.2f",
            "stats": _compute_stats(data),
        })
        matched.add(col)

    return kpis


def classify_correlation(r_value: float) -> tuple:
    """
    Classify correlation strength.

    Returns (label, is_meaningful) where is_meaningful is True only for |r| ≥ 0.3.
    Correlations below 0.3 are considered negligible or weak.
    """
    abs_r = abs(r_value)
    if abs_r < 0.1:
        return "negligible", False
    elif abs_r < 0.3:
        return "weak", False
    elif abs_r < 0.7:
        return "moderate", True
    elif abs_r < 0.9:
        return "strong", True
    else:
        return "very strong", True


def build_kpi_context(df: pd.DataFrame) -> str:
    """
    Build a domain-aware KPI context string for AI prompts.

    Used by 5_AI_Insights.py to provide accurate, domain-appropriate context
    instead of hard-coded e-commerce keyword matching.
    """
    domain_info = detect_domain(df)
    kpis = classify_kpis(df, domain_info)

    buf = io.StringIO()
    buf.write(f"--- Dataset Domain: {domain_info['label']} "
              f"(confidence: {domain_info['confidence']}) ---\n")
    if domain_info["matched_keywords"]:
        buf.write(f"Domain detected from columns: "
                  f"{', '.join(domain_info['matched_keywords'][:10])}\n")
    buf.write("\n")

    buf.write("--- Key Performance Indicators (Auto-Detected, Domain-Aware) ---\n")
    if not kpis:
        buf.write("No KPI columns detected.\n")
    else:
        for kpi in kpis:
            stats = kpi["stats"]
            line_parts = [f"{kpi['column']} [{kpi['label']}]"]

            if kpi["agg"] == "rate" and "rate" in stats:
                line_parts.append(f"Rate={stats['rate']:.1%}")
                line_parts.append(
                    f"({stats.get('count_positive', '?')}"
                    f"/{stats.get('count_total', '?')})"
                )
            else:
                if kpi["agg"] == "sum":
                    line_parts.append(f"Total={stats['sum']:,.2f}")
                line_parts.append(f"Mean={stats['mean']:,.2f}")
                line_parts.append(f"Median={stats['median']:,.2f}")
                line_parts.append(f"Min={stats['min']:,.2f}")
                line_parts.append(f"Max={stats['max']:,.2f}")
                if stats.get("std", 0) > 0:
                    line_parts.append(f"Std={stats['std']:,.2f}")

            buf.write("  " + ", ".join(line_parts) + "\n")

    buf.write("\n")
    return buf.getvalue()
