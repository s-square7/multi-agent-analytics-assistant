"""Data Scientist function tools: problem framing, features, risks, metrics, pipelines.

Deterministic helpers that turn a loose business goal into concrete ML guidance.
All return JSON strings for clean hand-off to the model.
"""

import json
from typing import List

from ._common import load_csv, make_crew_tool, safe_data_path, UnsafePathError

# Evaluation metrics keyed by problem type, with a short "when it matters" note.
_METRICS_BY_TYPE = {
    "classification": {
        "metrics": ["precision", "recall", "F1-score", "ROC-AUC", "PR-AUC"],
        "business_note": "Lean on recall when missing a positive case is costly (fraud, churn).",
    },
    "regression": {
        "metrics": ["MAE", "RMSE", "R2", "MAPE"],
        "business_note": "MAE/MAPE read naturally in business units.",
    },
    "forecasting": {
        "metrics": ["MAPE", "sMAPE", "RMSE", "MASE"],
        "business_note": "Backtest with rolling-origin evaluation and respect time order.",
    },
    "clustering": {
        "metrics": ["silhouette", "Davies-Bouldin", "Calinski-Harabasz"],
        "business_note": "Sanity-check clusters against business meaning, not just geometry.",
    },
    "anomaly_detection": {
        "metrics": ["precision@k", "recall", "PR-AUC", "alert_precision"],
        "business_note": "Tune the threshold to balance missed incidents against alert fatigue.",
    },
    "recommendation": {
        "metrics": ["precision@k", "recall@k", "NDCG", "MAP"],
        "business_note": "Judge ranking quality, not just point accuracy.",
    },
    "ranking": {
        "metrics": ["NDCG", "MRR", "MAP"],
        "business_note": "Position-aware metrics track the user experience.",
    },
}


def recommend_ml_problem_type(description: str) -> str:
    """Infer the ML problem type from a plain-language description of the goal."""
    t = (description or "").lower()
    if any(w in t for w in ["forecast", "next month", "time series", "predict demand", "seasonal"]):
        ptype, target = "forecasting", "future_value"
    elif any(w in t for w in ["churn", "fraud", "spam", "yes/no", "whether", "classify", "will they"]):
        ptype, target = "classification", "binary_label"
    elif any(w in t for w in ["how much", "amount", "value", "price", "revenue", "estimate"]):
        ptype, target = "regression", "numeric_value"
    elif any(w in t for w in ["segment", "group", "cluster", "persona"]):
        ptype, target = "clustering", None
    elif any(w in t for w in ["anomaly", "outlier", "unusual", "fraud detection"]):
        ptype, target = "anomaly_detection", "is_anomaly"
    elif any(w in t for w in ["recommend", "suggest items", "similar products"]):
        ptype, target = "recommendation", "relevance"
    else:
        ptype, target = "classification", "label"
    return json.dumps({
        "problem_type": ptype,
        "target_variable": target,
        "reason": f"Inferred '{ptype}' from the described objective.",
    })


def suggest_feature_engineering(data_context: str) -> str:
    """Suggest engineered features from a short description of the raw data."""
    t = (data_context or "").lower()
    features: List[str] = []
    if any(w in t for w in ["event", "session", "activity", "log"]):
        features += ["events_per_user_last_1_hour", "failed_event_ratio_last_24_hours",
                     "session_duration", "session_count_last_30_days"]
    if any(w in t for w in ["transaction", "purchase", "order", "revenue"]):
        features += ["avg_transaction_amount", "days_since_last_purchase", "rolling_7_day_revenue"]
    if any(w in t for w in ["customer", "user", "account"]):
        features += ["tenure_days", "support_ticket_count", "activity_recency"]
    if not features:
        features = ["temporal_aggregates", "categorical_frequency_encoding", "missingness_flags"]
    seen, out = set(), []
    for f in features:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return json.dumps({"feature_ideas": out})


def detect_ml_data_risks(csv_path: str) -> str:
    """Scan a sample CSV for modeling risks: imbalance, leakage hints, cardinality, dupes."""
    try:
        df = load_csv(csv_path)
    except UnsafePathError as err:
        return json.dumps({"error": str(err)})

    risks: List[str] = []
    if df.duplicated().any():
        risks.append("Duplicate records present — deduplicate before splitting.")
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        if 2 <= nun <= 3:
            frac = df[c].value_counts(normalize=True).max()
            if frac >= 0.8:
                risks.append(f"Potential class imbalance in '{c}' (majority {frac:.0%}).")
    for c in df.select_dtypes(include=["object"]).columns:
        if df[c].nunique() / max(1, len(df)) > 0.5 and not c.lower().endswith("_id"):
            risks.append(f"High-cardinality column '{c}' — needs an encoding strategy.")
    id_like = [c for c in df.columns if c.lower().endswith("_id") or c.lower() == "id"]
    if id_like:
        risks.append(f"ID columns {id_like} should stay out of features (leakage/overfit).")
    if any("date" in c.lower() or "time" in c.lower() for c in df.columns):
        risks.append("Time column present — use a time-based split, not a random one.")
    if df.isna().any().any():
        risks.append("Missing values present — decide imputation vs drop before training.")

    return json.dumps({"file": safe_data_path(csv_path).name, "risk_count": len(risks), "risks": risks})


def recommend_evaluation_metrics(problem_type: str) -> str:
    """Recommend evaluation metrics for a given ML problem type."""
    key = (problem_type or "").strip().lower()
    entry = _METRICS_BY_TYPE.get(
        key, {"metrics": ["accuracy", "F1-score"], "business_note": "Unknown type; showing defaults."})
    return json.dumps({"problem_type": key or "unknown", **entry})


def create_ml_pipeline_plan(problem_type: str) -> str:
    """Lay out an end-to-end ML pipeline for the given problem type."""
    key = (problem_type or "classification").strip().lower()
    split = "time-based split" if key == "forecasting" else "stratified train/test split"
    plan = [
        "Data ingestion from source CSV/warehouse",
        "Data validation and schema checks",
        "Feature engineering and transformation",
        f"Train-test split ({split})",
        "Model training with cross-validation",
        "Model evaluation against business metrics",
        "Model registry and versioning",
        "Batch or real-time inference service",
        "Monitoring (drift, latency, quality)",
        "Scheduled retraining and feedback loop",
    ]
    return json.dumps({"problem_type": key, "pipeline": plan})


def get_scientist_tools():
    """Return the data scientist tools wrapped for CrewAI."""
    return [
        make_crew_tool("Recommend ML Problem Type", recommend_ml_problem_type),
        make_crew_tool("Suggest Feature Engineering", suggest_feature_engineering),
        make_crew_tool("Detect ML Data Risks", detect_ml_data_risks),
        make_crew_tool("Recommend Evaluation Metrics", recommend_evaluation_metrics),
        make_crew_tool("Create ML Pipeline Plan", create_ml_pipeline_plan),
    ]
