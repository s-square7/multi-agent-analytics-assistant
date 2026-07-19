"""Data Analyst Agent builder."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from crewai import Agent

from function_tools.analyst_tools import get_analyst_tools


def build_data_analyst_agent(
    cfg: Dict[str, Any],
    llm: Any,
    extra_tools: Optional[List[Any]] = None,
    step_callback: Optional[Callable[[Any], None]] = None,
) -> Agent:
    """Build the Data Analyst Agent with analyst function tools (+ optional MCP tools)."""
    tools = list(get_analyst_tools())
    if extra_tools:
        tools.extend(extra_tools)
    return Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=llm,
        tools=tools,
        verbose=bool(cfg.get("verbose", True)),
        allow_delegation=bool(cfg.get("allow_delegation", False)),
        max_iter=int(cfg.get("max_iter", 5)),
        max_retry_limit=int(cfg.get("max_retry_limit", 2)),
        respect_context_window=True,
        use_system_prompt=True,
        step_callback=step_callback,
    )
