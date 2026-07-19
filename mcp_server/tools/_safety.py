"""Guardrails shared by the MCP analytics tools.

Everything the spec asks for in one place:
  - file access stays inside sample_data,
  - only known data extensions and a sane size limit,
  - SQL is read-only (no writes, no stacked statements),
  - plus small conveniences: a cached CSV reader and flexible list parsing.
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, List

import pandas as pd

# sample_data sits at mcp_server/sample_data (two levels up from this file).
SAMPLE_DATA_DIR = (Path(__file__).resolve().parent.parent / "sample_data").resolve()

ALLOWED_EXTENSIONS = {".csv", ".parquet", ".tsv"}
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB local read limit

# Statements that must never run against local data.
BLOCKED_SQL_KEYWORDS = (
    "DELETE", "UPDATE", "DROP", "ALTER", "INSERT",
    "MERGE", "TRUNCATE", "CREATE", "REPLACE", "GRANT",
    "REVOKE", "ATTACH", "COPY", "INSTALL", "LOAD",
)


class UnsafePathError(Exception):
    """Raised when a requested path escapes the sample_data sandbox."""


class UnsafeSQLError(Exception):
    """Raised when a SQL query is not read-only."""


def safe_resolve_data_path(csv_path: str) -> Path:
    """Resolve `csv_path` and guarantee it stays inside sample_data.

    Accepts a bare file name (`events_sample.csv`) or a path already inside the
    sample_data directory. Anything else raises UnsafePathError.
    """
    raw = Path(csv_path)
    candidate = raw if raw.is_absolute() or raw.parts[:1] == ("mcp_server",) else (SAMPLE_DATA_DIR / raw.name)
    resolved = candidate.resolve()

    if SAMPLE_DATA_DIR not in resolved.parents and resolved != SAMPLE_DATA_DIR:
        raise UnsafePathError(f"Access denied: '{csv_path}' is outside the sample_data folder.")
    if resolved.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise UnsafePathError(f"Unsupported file type '{resolved.suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}.")
    if not resolved.exists():
        raise UnsafePathError(f"File not found in sample_data: '{resolved.name}'.")
    if resolved.stat().st_size > MAX_FILE_BYTES:
        raise UnsafePathError("File exceeds the 50 MB local read limit.")
    return resolved


@lru_cache(maxsize=16)
def _read_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path_str)


def load_csv(csv_path: str) -> pd.DataFrame:
    """Read a sandboxed CSV once and reuse it (cached by path + mtime). Read-only."""
    path = safe_resolve_data_path(csv_path)
    return _read_csv_cached(str(path), path.stat().st_mtime)


def as_list(value: Any) -> List[str]:
    """Accept a list, a JSON-list string, or a comma string; return a list of strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in text.split(",") if part.strip()]
    return [str(value).strip()]


def assert_read_only_sql(query: str) -> None:
    """Raise UnsafeSQLError unless `query` is a single read-only statement."""
    if not query or not query.strip():
        raise UnsafeSQLError("Empty SQL query.")

    # Drop comments before scanning for keywords.
    stripped = re.sub(r"--.*?$", " ", query, flags=re.MULTILINE)
    stripped = re.sub(r"/\*.*?\*/", " ", stripped, flags=re.DOTALL)
    upper = stripped.upper()

    for word in BLOCKED_SQL_KEYWORDS:
        if re.search(rf"\b{word}\b", upper):
            raise UnsafeSQLError(f"Blocked SQL keyword detected: {word}.")

    first_token = upper.strip().split(None, 1)[0] if upper.strip() else ""
    if first_token not in ("SELECT", "WITH"):
        raise UnsafeSQLError("Only read-only SELECT/WITH queries are allowed.")

    if ";" in query.strip().rstrip(";"):
        raise UnsafeSQLError("Multiple SQL statements are not allowed.")
