"""Safe SQL validation + DuckDB execution MCP tools."""


import re
from typing import Any, Dict

import duckdb
import sqlglot
from sqlglot import expressions as exp

from ._safety import (
    assert_read_only_sql,
    safe_resolve_data_path,
    UnsafeSQLError,
)


def validate_sql(query: str, has_partition_column: bool = False) -> Dict[str, Any]:
    """Validate that a SQL query is read-only and follows good hygiene."""
    warnings = []
    errors = []
    read_only = True

    try:
        assert_read_only_sql(query)
    except UnsafeSQLError as err:
        read_only = False
        errors.append(str(err))

    # Best-effort parse for structural hygiene checks.
    try:
        parsed = sqlglot.parse_one(query, read="duckdb")
    except Exception:  # noqa: BLE001 - parsing is advisory only
        parsed = None
        warnings.append("Query could not be fully parsed; structural checks skipped.")

    if parsed is not None:
        if any(isinstance(s, exp.Star) for s in parsed.find_all(exp.Star)):
            warnings.append("Avoid SELECT * — list explicit columns instead.")
        if parsed.find(exp.Limit) is None:
            warnings.append("No LIMIT clause — add one to cap result size.")
        where = parsed.find(exp.Where)
        has_date_filter = bool(where) and bool(
            re.search(r"date|time", where.sql(), flags=re.IGNORECASE)
        )
        if not has_date_filter:
            warnings.append("No date filter detected — event queries should be time-bounded.")
        if has_partition_column and not has_date_filter:
            warnings.append("Partition column available but not used in WHERE.")

    return {
        "is_read_only": read_only,
        "is_safe": read_only and not errors,
        "errors": errors,
        "warnings": warnings,
    }


def run_duckdb_query(query: str, csv_path: str, limit: int = 100) -> Dict[str, Any]:
    """Run a read-only SQL query against a sample CSV using DuckDB.

    The CSV is exposed to the query as the table name ``data``.
    """
    verdict = validate_sql(query)
    if not verdict["is_safe"]:
        return {"ok": False, "errors": verdict["errors"], "rows": []}

    path = safe_resolve_data_path(csv_path)
    # Path is already sandboxed to sample_data; escape quotes defensively.
    safe_path = str(path).replace("'", "''")
    con = duckdb.connect(database=":memory:")
    try:
        con.execute(
            f"CREATE VIEW data AS SELECT * FROM read_csv_auto('{safe_path}', header=true)"
        )
        capped = f"SELECT * FROM ({query.rstrip(';')}) AS _q LIMIT {int(limit)}"
        cursor = con.execute(capped)
        cols = [d[0] for d in cursor.description]
        records = [dict(zip(cols, row)) for row in cursor.fetchall()]
        return {"ok": True, "columns": cols, "row_count": len(records), "rows": records}
    except Exception as err:  # noqa: BLE001 - surface a clean message, not a stack trace
        return {"ok": False, "errors": [f"Query execution failed: {err}"], "rows": []}
    finally:
        con.close()
