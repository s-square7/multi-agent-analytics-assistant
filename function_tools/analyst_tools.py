"""Data Analyst function tools: profiling, KPIs, dashboards, SQL safety, explanations.

These are deterministic Python helpers wrapped as CrewAI tools. They return JSON
strings so the model gets clean, structured results it can quote back.
"""

import json
import re
from typing import List

from ._common import as_list, load_csv, make_crew_tool, safe_data_path, UnsafePathError

# Verbs that must never appear in a query the analyst calls "safe".
_BLOCKED_SQL = ("DELETE", "UPDATE", "DROP", "ALTER", "INSERT", "MERGE", "TRUNCATE", "CREATE")


def profile_dataframe(csv_path: str) -> str:
    """Profile a sample CSV: shape, dtypes, missing values, duplicates, and a few rows."""
    try:
        df = load_csv(csv_path)
    except UnsafePathError as err:
        return json.dumps({"error": str(err)})
    return json.dumps({
        "file": safe_data_path(csv_path).name,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "data_types": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()},
        "duplicate_rows": int(df.duplicated().sum()),
        "sample_records": df.head(3).fillna("").astype(str).to_dict(orient="records"),
    })


def suggest_kpi_metrics(domain: str, columns: str) -> str:
    """Suggest KPIs from a business domain and dataset columns.

    `columns` may be a list, a JSON list, or a comma-separated string.
    """
    cols = {c.lower() for c in as_list(columns)}
    kpis: List[str] = []
    d = (domain or "").lower()
    if "ecommerce" in d or {"revenue", "order_id"} & cols:
        kpis += ["Total revenue", "Average order value", "Repeat purchase rate",
                 "Order cancellation rate", "Monthly active customers"]
    if "event" in d or {"event_type", "session_id"} & cols:
        kpis += ["Active sessions", "Error rate", "Average session duration", "Checkout rate"]
    if not kpis:
        kpis = ["Record volume", "Distinct entities", "Completeness rate"]
    seen, out = set(), []
    for k in kpis:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return json.dumps({"domain": d or "generic", "recommended_kpis": out})


def generate_dashboard_layout(domain: str, columns: str) -> str:
    """Propose a dashboard structure (sections, chart types, filters) for the data."""
    cols = as_list(columns)
    time_cols = [c for c in cols if re.search(r"date|time", c, re.I)]
    cat_cols = [c for c in cols if re.search(r"status|type|segment|channel|country", c, re.I)]
    return json.dumps({
        "dashboard_name": f"{(domain or 'Business').title()} Overview Dashboard",
        "sections": ["KPI Overview", "Trends Over Time", "Segment Breakdown", "Quality Indicators"],
        "chart_types": ["kpi_cards", "line_chart", "bar_chart", "table"],
        "filters": (time_cols[:1] + cat_cols[:2]) or ["date"],
        "drill_downs": cat_cols[:2] or ["segment"],
    })


def validate_sql_safety(query: str) -> str:
    """Check whether a SQL query is safe: read-only, plus a few hygiene warnings."""
    q = query or ""
    upper = q.upper()
    blocked = [w for w in _BLOCKED_SQL if re.search(rf"\b{w}\b", upper)]
    warnings = []
    if re.search(r"SELECT\s+\*", upper):
        warnings.append("Avoid SELECT * — list explicit columns.")
    if not re.search(r"\bWHERE\b.*(DATE|TIME)", upper):
        warnings.append("No date filter — event queries should be time-bounded.")
    if "LIMIT" not in upper:
        warnings.append("No LIMIT clause — cap result size.")
    is_select = upper.strip().startswith(("SELECT", "WITH"))
    return json.dumps({
        "is_safe": bool(is_select and not blocked),
        "is_select_only": is_select,
        "blocked_keywords": blocked,
        "warnings": warnings,
    })


def explain_query_result(metric: str, trend: str, change_percent: float) -> str:
    """Turn a metric, its trend, and a percent change into a plain-language read-out."""
    try:
        change = float(change_percent)
    except (TypeError, ValueError):
        change = 0.0
    direction = "increased" if change >= 0 else "decreased"
    m = (metric or "the metric").replace("_", " ")
    note = ""
    if "revenue" in m.lower() and change < 0:
        note = " This may point to weaker acquisition, fewer repeat purchases, or a seasonal dip."
    elif "error" in m.lower() and change > 0:
        note = " Rising errors can signal a reliability regression — check recent releases."
    return f"{m.title()} {direction} by {abs(change):.1f}% (trend: {trend}).{note}"


def get_analyst_tools():
    """Return the analyst tools wrapped for CrewAI."""
    return [
        make_crew_tool("Profile Dataframe", profile_dataframe),
        make_crew_tool("Suggest KPI Metrics", suggest_kpi_metrics),
        make_crew_tool("Generate Dashboard Layout", generate_dashboard_layout),
        make_crew_tool("Validate SQL Safety", validate_sql_safety),
        make_crew_tool("Explain Query Result", explain_query_result),
    ]
