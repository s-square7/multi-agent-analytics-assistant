"""CSV profiling and data-dictionary MCP tools (pandas-backed)."""

from typing import Any, Dict

from ._safety import load_csv, safe_resolve_data_path


def profile_csv(csv_path: str) -> Dict[str, Any]:
    """Read a CSV from sample_data and return a structured profile."""
    df = load_csv(csv_path)
    missing = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    return {
        "file": safe_resolve_data_path(csv_path).name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "data_types": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": missing,
        "duplicates": int(df.duplicated().sum()),
        "sample_rows": df.head(3).fillna("").astype(str).to_dict(orient="records"),
    }


def _guess_meaning(name: str) -> str:
    n = name.lower()
    if n.endswith("_id") or n == "id":
        return "Unique identifier"
    if "date" in n or "time" in n:
        return "Timestamp / date field"
    if any(k in n for k in ("revenue", "amount", "value", "price")):
        return "Monetary measure"
    if any(k in n for k in ("status", "type", "segment", "channel")):
        return "Categorical attribute"
    if n.startswith(("is_", "has_")):
        return "Boolean flag"
    if any(k in n for k in ("count", "duration", "qty")):
        return "Numeric measure"
    return "General attribute"


def create_data_dictionary(csv_path: str) -> Dict[str, Any]:
    """Build a simple data dictionary: name, type, likely meaning, and sample values."""
    df = load_csv(csv_path).head(5000)
    columns = []
    for c in df.columns:
        non_null = df[c].dropna()
        columns.append({
            "name": c,
            "type": str(df[c].dtype),
            "possible_meaning": _guess_meaning(c),
            "distinct_values": int(non_null.nunique()),
            "sample_values": [str(v) for v in non_null.unique()[:3]],
        })
    return {"file": safe_resolve_data_path(csv_path).name, "columns": columns}
