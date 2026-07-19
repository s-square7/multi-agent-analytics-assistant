"""Supervisor Agent builder (hierarchical manager).

IMPORTANT: In CrewAI's hierarchical process the manager_agent must NOT have any
tools attached — CrewAI raises "Manager agent should not have tools" otherwise.
The Supervisor therefore orchestrates and delegates tool-free; the Data Analyst
and Data Scientist workers carry all the function and MCP tools.
"""

from typing import Any, Callable, Dict, List, Optional

from crewai import Agent


def build_supervisor_agent(
    cfg: Dict[str, Any],
    llm: Any,
    extra_tools: Optional[List[Any]] = None,  # accepted but intentionally unused (manager stays tool-free)
    step_callback: Optional[Callable[[Any], None]] = None,
) -> Agent:
    """Build the Supervisor Agent (hierarchical manager) with NO tools."""
    return Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=llm,
        # No tools: required for a CrewAI hierarchical manager_agent.
        verbose=bool(cfg.get("verbose", True)),
        allow_delegation=bool(cfg.get("allow_delegation", True)),
        max_iter=int(cfg.get("max_iter", 8)),
        max_retry_limit=int(cfg.get("max_retry_limit", 2)),
        respect_context_window=True,
        use_system_prompt=True,
        step_callback=step_callback,
    )
