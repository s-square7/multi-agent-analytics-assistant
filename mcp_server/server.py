"""analytics_mcp_server — a local MCP server of reusable analytics tools.

Run it standalone over stdio with:

    python mcp_server/server.py

CrewAI agents connect through crewai_tools.MCPServerAdapter. Each tool here is a
thin wrapper over the plain functions in mcp_server/tools, so the same logic is
unit-tested directly without a running transport.

Column arguments are typed as plain strings on purpose: local models reliably
pass "revenue, order_id" but often stumble on strict JSON arrays. The underlying
helpers accept a list, a JSON list, or a comma string, so both styles work.
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Allow running both as `python mcp_server/server.py` and as an imported module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from tools import csv_profile_tools  # noqa: E402
from tools import sql_tools  # noqa: E402
from tools import data_quality_tools  # noqa: E402
from tools import kpi_tools  # noqa: E402
from tools import ml_tools  # noqa: E402
from tools import report_tools  # noqa: E402

mcp = FastMCP("analytics_mcp_server")


@mcp.tool()
def mcp_profile_csv(csv_path: str) -> Dict[str, Any]:
    """Profile a CSV in sample_data: rows, columns, dtypes, missing values, duplicates, sample rows."""
    return csv_profile_tools.profile_csv(csv_path)


@mcp.tool()
def mcp_create_data_dictionary(csv_path: str) -> Dict[str, Any]:
    """Build a data dictionary (column name, type, likely meaning, sample values) for a sample CSV."""
    return csv_profile_tools.create_data_dictionary(csv_path)


@mcp.tool()
def mcp_validate_sql(query: str, has_partition_column: bool = False) -> Dict[str, Any]:
    """Validate that a SQL query is read-only and follows hygiene rules (LIMIT, no SELECT *, date filter)."""
    return sql_tools.validate_sql(query, has_partition_column=has_partition_column)


@mcp.tool()
def mcp_run_duckdb_query(query: str, csv_path: str, limit: int = 100) -> Dict[str, Any]:
    """Run a read-only SQL SELECT against a sample CSV via DuckDB. The CSV is exposed as table `data`."""
    return sql_tools.run_duckdb_query(query, csv_path, limit=limit)


@mcp.tool()
def mcp_detect_data_quality_issues(csv_path: str) -> Dict[str, Any]:
    """Detect data-quality problems: missing values, duplicates, constants, high cardinality, negatives, outliers."""
    return data_quality_tools.detect_data_quality_issues(csv_path)


@mcp.tool()
def mcp_generate_kpi_catalog(domain: str, columns: str = "") -> Dict[str, Any]:
    """Build a KPI catalog (name, formula, grain, business use) from a domain and dataset columns."""
    return kpi_tools.generate_kpi_catalog(domain, columns)


@mcp.tool()
def mcp_recommend_ml_use_cases(columns: str) -> Dict[str, Any]:
    """Recommend ML use cases (problem type, required columns, business value) from dataset columns."""
    return ml_tools.recommend_ml_use_cases(columns)


@mcp.tool()
def mcp_feature_engineering_suggestions(columns: str) -> Dict[str, Any]:
    """Suggest feature-engineering ideas for event, customer, transaction, or time-series columns."""
    return ml_tools.feature_engineering_suggestions(columns)


@mcp.tool()
def mcp_anomaly_detection_summary(csv_path: str, column: str = "", method: str = "zscore") -> Dict[str, Any]:
    """Summarise anomalies in a numeric column. method = zscore | iqr | isolation_forest."""
    return ml_tools.anomaly_detection_summary(csv_path, column=column or None, method=method)


@mcp.tool()
def mcp_generate_report_markdown(sections: Dict[str, Any]) -> str:
    """Combine tool outputs into a final markdown report with the standard analytics sections."""
    return report_tools.generate_report_markdown(sections)


if __name__ == "__main__":
    mcp.run(transport="stdio")
