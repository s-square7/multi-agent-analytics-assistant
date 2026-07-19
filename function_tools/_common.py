"""Small shared helpers for the local function tools.

Three things live here so every tool behaves consistently:
  - safe_data_path: keep file access inside the sample_data folder,
  - load_csv:       read a CSV once and reuse it (cached by file + mtime),
  - as_list:        accept a real list, a JSON list, or a comma string.

The last one matters a lot with small local models: they often hand a tool
`"revenue, order_id"` instead of `["revenue", "order_id"]`, and we don't want
that to crash a request.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, List

import pandas as pd

# Function tools may only read the datasets we ship.
SAMPLE_DATA_DIR = (
    Path(__file__).resolve().parent.parent / "mcp_server" / "sample_data"
).resolve()

ALLOWED_EXTENSIONS = {".csv", ".tsv", ".parquet"}


class UnsafePathError(Exception):
    """Raised when a requested path points outside sample_data."""


def safe_data_path(csv_path: str) -> Path:
    """Resolve a data path and confine it to the sample_data folder."""
    raw = Path(csv_path)
    candidate = raw if raw.is_absolute() else (SAMPLE_DATA_DIR / raw.name)
    resolved = candidate.resolve()
    if SAMPLE_DATA_DIR not in resolved.parents and resolved != SAMPLE_DATA_DIR:
        raise UnsafePathError(f"Access denied outside sample_data: '{csv_path}'.")
    if resolved.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise UnsafePathError(f"Unsupported file type: '{resolved.suffix}'.")
    if not resolved.exists():
        raise UnsafePathError(f"File not found in sample_data: '{resolved.name}'.")
    return resolved


@lru_cache(maxsize=16)
def _read_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    # Keyed on mtime so an edited file is re-read automatically.
    return pd.read_csv(path_str)


def load_csv(csv_path: str) -> pd.DataFrame:
    """Read a sandboxed CSV, reusing the parsed copy when possible.

    Treat the returned frame as read-only; tools here only compute over it.
    """
    path = safe_data_path(csv_path)
    return _read_csv_cached(str(path), path.stat().st_mtime)


def as_list(value: Any) -> List[str]:
    """Turn a list, a JSON-list string, or a comma string into a list of strings."""
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


def make_crew_tool(name: str, func):
    """Wrap a plain function as a CrewAI tool (imported lazily so tests need no crewai)."""
    from crewai.tools import tool

    return tool(name)(func)
