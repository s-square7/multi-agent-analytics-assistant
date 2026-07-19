"""ML-oriented MCP tools: use-case mining, feature ideas, and anomaly summaries."""

from typing import Any, Dict, List

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

from ._safety import as_list, load_csv, safe_resolve_data_path


def recommend_ml_use_cases(columns) -> Dict[str, Any]:
    """Suggest ML use cases from dataset columns (list/JSON/comma all accepted)."""
    cols = {c.lower() for c in as_list(columns)}
    use_cases: List[Dict[str, Any]] = []

    if any("churn" in c for c in cols) or {"is_active", "customer_id"} <= cols:
        use_cases.append({
            "use_case": "churn prediction", "problem_type": "classification",
            "required_columns": ["customer_id", "is_active"],
            "business_value": "Reduce customer loss by targeting at-risk accounts.",
        })
    if "revenue" in cols and ({"order_date"} & cols or {"event_time"} & cols):
        use_cases.append({
            "use_case": "revenue forecasting", "problem_type": "forecasting",
            "required_columns": ["order_date", "revenue"],
            "business_value": "Plan inventory and budgets from projected demand.",
        })
    if {"lifetime_value", "customer_id"} <= cols:
        use_cases.append({
            "use_case": "customer lifetime value regression", "problem_type": "regression",
            "required_columns": ["customer_id", "lifetime_value"],
            "business_value": "Prioritise high-value customer acquisition.",
        })
    if {"event_type", "duration_ms"} <= cols:
        use_cases.append({
            "use_case": "anomaly detection on engagement", "problem_type": "anomaly_detection",
            "required_columns": ["event_time", "duration_ms"],
            "business_value": "Catch outages and abuse from abnormal behaviour.",
        })
    if {"segment", "customer_id"} <= cols:
        use_cases.append({
            "use_case": "customer segmentation", "problem_type": "clustering",
            "required_columns": ["customer_id", "segment", "lifetime_value"],
            "business_value": "Tailor marketing to behavioural clusters.",
        })

    if not use_cases:
        use_cases.append({
            "use_case": "exploratory baseline modeling", "problem_type": "classification",
            "required_columns": list(cols)[:4],
            "business_value": "Establish a baseline before scoping richer use cases.",
        })
    return {"use_case_count": len(use_cases), "ml_use_cases": use_cases}


def feature_engineering_suggestions(columns) -> Dict[str, Any]:
    """Suggest engineered features for the given columns."""
    cols = {c.lower() for c in as_list(columns)}
    features: List[str] = []
    if {"event_time"} & cols or {"order_date"} & cols:
        features += ["events_per_user_last_1_hour", "failed_event_ratio_last_24_hours",
                     "rolling_7_day_revenue", "days_since_last_activity", "hour_of_day", "day_of_week"]
    if "revenue" in cols:
        features += ["avg_transaction_amount", "revenue_trend_30d", "refund_ratio"]
    if "session_id" in cols:
        features += ["session_count_last_30_days", "avg_session_duration"]
    if "customer_id" in cols:
        features += ["tenure_days", "lifetime_event_count"]
    if not features:
        features = ["row_level_aggregates", "categorical_frequency_encoding", "missingness_flags"]
    seen, out = set(), []
    for f in features:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return {"feature_count": len(out), "features": out}


def anomaly_detection_summary(csv_path: str, column: str = None, method: str = "zscore") -> Dict[str, Any]:
    """Summarise anomalies in a numeric column. method = zscore | iqr | isolation_forest."""
    df = load_csv(csv_path)
    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    if not numeric_cols:
        return {"ok": False, "error": "No numeric columns available for anomaly detection."}

    target = column if (column and column in numeric_cols) else numeric_cols[0]
    series = df[target].dropna().astype(float)
    if len(series) < 10:
        return {"ok": False, "error": f"Not enough data in '{target}' for anomaly detection."}

    values = series.to_numpy()
    if method == "iqr":
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (values < lo) | (values > hi)
    elif method == "isolation_forest":
        preds = IsolationForest(contamination="auto", random_state=42).fit_predict(values.reshape(-1, 1))
        mask = preds == -1
    else:
        method = "zscore"
        mask = np.abs(stats.zscore(values)) > 3.0

    n_anom = int(mask.sum())
    return {
        "ok": True,
        "file": safe_resolve_data_path(csv_path).name,
        "column": target,
        "method": method,
        "total_points": int(len(values)),
        "anomaly_count": n_anom,
        "anomaly_ratio": round(n_anom / len(values), 4),
        "examples": [round(float(v), 2) for v in values[mask][:5]],
    }
