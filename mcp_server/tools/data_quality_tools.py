"""Data-quality detection MCP tool (pandas-backed)."""

from typing import Any, Dict, List

from ._safety import load_csv, safe_resolve_data_path

# Columns we expect to hold only non-negative values.
POSITIVE_ONLY_HINTS = ("revenue", "amount", "price", "value", "qty", "count", "duration")


def detect_data_quality_issues(csv_path: str) -> Dict[str, Any]:
    """Flag common data-quality problems and give a rough 0-100 quality score."""
    df = load_csv(csv_path)
    issues: List[Dict[str, Any]] = []

    missing = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    if missing:
        issues.append({"type": "missing_values", "severity": "medium", "detail": missing})

    dup = int(df.duplicated().sum())
    if dup:
        issues.append({"type": "duplicate_rows", "severity": "medium", "detail": {"count": dup}})

    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant:
        issues.append({"type": "constant_columns", "severity": "low", "detail": constant})

    high_card = []
    for c in df.select_dtypes(include=["object"]).columns:
        ratio = df[c].nunique(dropna=True) / max(1, len(df))
        if ratio > 0.5 and not c.lower().endswith("_id"):
            high_card.append({"column": c, "distinct_ratio": round(ratio, 3)})
    if high_card:
        issues.append({"type": "high_cardinality", "severity": "low", "detail": high_card})

    negatives = {}
    for c in df.select_dtypes(include=["number"]).columns:
        if any(h in c.lower() for h in POSITIVE_ONLY_HINTS):
            n = int((df[c] < 0).sum())
            if n:
                negatives[c] = n
    if negatives:
        issues.append({"type": "negative_values", "severity": "high", "detail": negatives})

    outliers = {}
    for c in df.select_dtypes(include=["number"]).columns:
        s = df[c].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < lo) | (s > hi)).sum())
        if n:
            outliers[c] = n
    if outliers:
        issues.append({"type": "outliers_iqr", "severity": "low", "detail": outliers})

    score = max(0, 100 - 8 * len(issues) - min(20, dup) - 3 * len(missing))
    return {
        "file": safe_resolve_data_path(csv_path).name,
        "rows": int(len(df)),
        "issue_count": len(issues),
        "quality_score": int(score),
        "issues": issues,
    }
