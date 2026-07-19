"""Bridge between CrewAI agents and the local analytics MCP server.

Uses crewai_tools.MCPServerAdapter over stdio to launch mcp_server/server.py and
expose its tools to the agents. Everything is optional and defensive: if the MCP
SDK or adapter is unavailable, the caller falls back to local function tools only.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent
SERVER_PATH = BASE_DIR / "mcp_server" / "server.py"

# Which agents may use which MCP tools (per project spec section 8).
MCP_TOOL_ROUTING: Dict[str, List[str]] = {
    "supervisor": [
        "mcp_validate_sql", "mcp_generate_kpi_catalog", "mcp_recommend_ml_use_cases",
        "mcp_create_data_dictionary", "mcp_generate_report_markdown",
    ],
    "analyst": [
        "mcp_profile_csv", "mcp_run_duckdb_query", "mcp_validate_sql",
        "mcp_detect_data_quality_issues", "mcp_generate_kpi_catalog",
        "mcp_anomaly_detection_summary", "mcp_create_data_dictionary",
    ],
    "scientist": [
        "mcp_profile_csv", "mcp_detect_data_quality_issues", "mcp_recommend_ml_use_cases",
        "mcp_feature_engineering_suggestions", "mcp_anomaly_detection_summary",
    ],
}


def mcp_available() -> bool:
    """True if the MCP SDK and CrewAI adapter appear importable."""
    try:
        import mcp  # noqa: F401
        from crewai_tools import MCPServerAdapter  # noqa: F401
        return SERVER_PATH.exists()
    except Exception:  # noqa: BLE001
        return False


def _tool_name(tool: Any) -> str:
    return getattr(tool, "name", "") or getattr(tool, "__name__", "") or str(tool)


def _distribute(tools: List[Any]) -> Dict[str, List[Any]]:
    """Split the flat MCP tool list into per-agent buckets by tool name."""
    by_agent: Dict[str, List[Any]] = {"supervisor": [], "analyst": [], "scientist": []}
    for tool in tools:
        name = _tool_name(tool)
        for agent_key, allowed in MCP_TOOL_ROUTING.items():
            if any(name == a or name.endswith(a) for a in allowed):
                by_agent[agent_key].append(tool)
    return by_agent


@contextmanager
def mcp_tools_by_agent():
    """Yield a dict of per-agent MCP tools. Raises if MCP cannot be started.

    Must wrap the crew kickoff — the stdio connection stays open for the duration.
    """
    from crewai_tools import MCPServerAdapter
    from mcp import StdioServerParameters

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
        env={**os.environ, "PYTHONPATH": str(BASE_DIR)},
    )
    with MCPServerAdapter(params) as tools:
        yield _distribute(list(tools))
