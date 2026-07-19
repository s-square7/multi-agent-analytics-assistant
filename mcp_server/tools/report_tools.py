"""Markdown report assembly MCP tool."""


import json
from typing import Any, Dict


def _fmt(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return "```json\n" + json.dumps(value, indent=2, default=str) + "\n```"


def generate_report_markdown(sections: Dict[str, Any]) -> str:
    """Combine tool outputs into a final markdown analytics report.

    ``sections`` is a mapping of section title -> content (string or JSON-able).
    Missing sections are rendered with a placeholder so the structure is stable.
    """
    order = [
        "Dataset Summary",
        "Data Quality Findings",
        "Recommended KPIs",
        "ML Use Cases",
        "Risks",
        "Next Steps",
    ]
    lines = ["# Analytics Report", ""]
    for title in order:
        lines.append(f"## {title}")
        if title in sections and sections[title] not in (None, "", {}, []):
            lines.append(_fmt(sections[title]))
        else:
            lines.append("_No data provided for this section._")
        lines.append("")
    # Any extra sections supplied beyond the standard set.
    for title, content in sections.items():
        if title not in order:
            lines.append(f"## {title}")
            lines.append(_fmt(content))
            lines.append("")
    return "\n".join(lines).strip()
