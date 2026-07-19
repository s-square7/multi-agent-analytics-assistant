"""Supervisor function tools: routing, planning, history, validation, context.

These are deterministic (no LLM calls), so the crew has stable, predictable
primitives for classifying a request and checking the final answer. Each returns
a string — JSON where there's structure — to stay easy for the model to read.
"""


import json
from typing import List

from ._common import make_crew_tool

# Keyword -> intent map for request classification.
_INTENT_KEYWORDS = {
    "dashboard": ["dashboard", "kpi", "chart", "visual", "report card", "bi", "tableau", "power bi"],
    "sql": ["sql", "query", "duckdb", "join", "select", "table"],
    "data_quality": ["quality", "missing", "duplicate", "null", "clean", "validate", "outlier"],
    "data_science": ["ml", "model", "predict", "forecast", "classification", "regression",
                     "cluster", "feature", "train", "churn", "anomaly"],
    "analytics": ["kpi", "metric", "trend", "insight", "eda", "profile", "analy"],
    "architecture": ["architecture", "pipeline", "deploy", "scal", "stream", "ingestion", "design"],
}

_AGENT_FOR_INTENT = {
    "dashboard": "Data Analyst Agent",
    "sql": "Data Analyst Agent",
    "data_quality": "Data Analyst Agent",
    "analytics": "Data Analyst Agent",
    "data_science": "Data Scientist Agent",
    "architecture": "Data Scientist Agent",
    "mixed": "Both Agents",
}


def classify_user_request(user_request: str) -> str:
    """Classify a user request into an analytics intent and recommend an agent.

    Returns JSON with keys: intent, recommended_agent, reason.
    """
    text = (user_request or "").lower()
    scores = {intent: sum(1 for k in kws if k in text) for intent, kws in _INTENT_KEYWORDS.items()}
    hit = {i: s for i, s in scores.items() if s > 0}
    if not hit:
        intent = "analytics"
    else:
        top = sorted(hit.items(), key=lambda kv: kv[1], reverse=True)
        analyst_side = {"dashboard", "sql", "data_quality", "analytics"}
        science_side = {"data_science", "architecture"}
        has_a = any(i in analyst_side for i, _ in top)
        has_s = any(i in science_side for i, _ in top)
        intent = "mixed" if (has_a and has_s) else top[0][0]
    return json.dumps({
        "intent": intent,
        "recommended_agent": _AGENT_FOR_INTENT.get(intent, "Data Analyst Agent"),
        "reason": f"Matched intent '{intent}' from request keywords.",
    })


def create_agent_work_plan(user_request: str) -> str:
    """Produce a step-by-step delegation work plan. Returns JSON with a 'steps' list."""
    intent = json.loads(classify_user_request(user_request))["intent"]
    steps: List[str] = []
    if intent in ("analytics", "data_quality", "dashboard", "sql", "mixed"):
        steps.append("Ask Data Analyst Agent to profile the dataset and flag quality issues.")
        steps.append("Ask Data Analyst Agent to suggest KPIs and a dashboard layout.")
    if intent in ("data_science", "architecture", "mixed"):
        steps.append("Ask Data Scientist Agent to identify ML use cases and feature ideas.")
        steps.append("Ask Data Scientist Agent to outline an ML pipeline and evaluation metrics.")
    steps.append("Combine specialist outputs into one final structured answer.")
    return json.dumps({"intent": intent, "steps": steps})


def summarize_chat_history(chat_history: str) -> str:
    """Compress prior conversation into a short extractive summary to save context."""
    if not chat_history or not chat_history.strip():
        return "No prior conversation."
    lines = [ln.strip() for ln in chat_history.splitlines() if ln.strip()]
    user_lines = [ln for ln in lines if ln.upper().startswith("USER")]
    topics = set()
    for kw in ("dashboard", "kpi", "sql", "ml", "forecast", "churn", "quality",
               "pipeline", "csv", "mcp", "revenue", "anomaly"):
        if any(kw in ln.lower() for ln in lines):
            topics.add(kw)
    recent = user_lines[-2:] if user_lines else lines[-2:]
    summary = "Recent user focus: " + " | ".join(recent)
    if topics:
        summary += f"\nTopics seen: {', '.join(sorted(topics))}."
    return summary[:800]


_REQUIRED_SECTIONS = [
    "Direct Answer", "Dataset Summary", "Data Quality", "Recommended KPIs",
    "ML Use Cases", "Risks", "Next Steps",
]


def validate_final_response_structure(response: str) -> str:
    """Check whether a final response contains the required sections. Returns JSON."""
    text = (response or "").lower()
    present, missing = [], []
    for section in _REQUIRED_SECTIONS:
        (present if section.lower() in text else missing).append(section)
    return json.dumps({
        "is_complete": not missing,
        "present_sections": present,
        "missing_sections": missing,
    })


def estimate_context_usage(text: str, context_window: int = 8192) -> str:
    """Estimate context-window usage for a block of text (~4 chars/token). Returns JSON."""
    tokens = max(1, len(text or "") // 4)
    pct = round(tokens / max(1, context_window) * 100, 2)
    return json.dumps({
        "estimated_input_tokens": tokens,
        "context_window": context_window,
        "usage_percent": pct,
    })


def get_supervisor_tools():
    """Return CrewAI-wrapped supervisor tools."""
    return [
        make_crew_tool("Classify User Request", classify_user_request),
        make_crew_tool("Create Agent Work Plan", create_agent_work_plan),
        make_crew_tool("Summarize Chat History", summarize_chat_history),
        make_crew_tool("Validate Final Response Structure", validate_final_response_structure),
        make_crew_tool("Estimate Context Usage", estimate_context_usage),
    ]
