"""Multi-Agent Analytics Assistant — CrewAI + Ollama + MCP + Streamlit.

Supervisor Agent (hierarchical manager) delegates to a Data Analyst Agent and a
Data Scientist Agent. Each specialist has local function tools and, when available,
tools from the local analytics MCP server. The UI shows a live activity timeline,
context/usage metrics, and a delegation trace.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
import streamlit as st
import yaml
from crewai import Crew, LLM, Process

from agents.supervisor_agent import build_supervisor_agent
from agents.data_analyst_agent import build_data_analyst_agent
from agents.data_scientist_agent import build_data_scientist_agent
from mcp_integration import mcp_available, mcp_tools_by_agent

# =========================================================
# Streamlit Config
# =========================================================

st.set_page_config(page_title="Multi-Agent Analytics Assistant", layout="wide")

BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
AGENTS_YAML_PATH = CONFIG_DIR / "agents.yaml"
TASKS_YAML_PATH = CONFIG_DIR / "tasks.yaml"

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL_NAME = "llama3.2:3b"

# Static catalogs (for sidebar display only).
FUNCTION_TOOL_CATALOG: Dict[str, List[str]] = {
    "Supervisor Agent": [
        "classify_user_request", "create_agent_work_plan", "summarize_chat_history",
        "validate_final_response_structure", "estimate_context_usage",
    ],
    "Data Analyst Agent": [
        "profile_dataframe", "suggest_kpi_metrics", "generate_dashboard_layout",
        "validate_sql_safety", "explain_query_result",
    ],
    "Data Scientist Agent": [
        "recommend_ml_problem_type", "suggest_feature_engineering", "detect_ml_data_risks",
        "recommend_evaluation_metrics", "create_ml_pipeline_plan",
    ],
}

MCP_TOOL_CATALOG: List[str] = [
    "mcp_profile_csv", "mcp_run_duckdb_query", "mcp_validate_sql",
    "mcp_detect_data_quality_issues", "mcp_generate_kpi_catalog",
    "mcp_recommend_ml_use_cases", "mcp_feature_engineering_suggestions",
    "mcp_anomaly_detection_summary", "mcp_create_data_dictionary",
    "mcp_generate_report_markdown",
]


# =========================================================
# Event Timeline Helpers
# =========================================================

def now_time() -> str:
    return datetime.now().strftime("%H:%M:%S")


def add_event(events: List[Dict[str, Any]], event_type: str, title: str,
              detail: str = "", agent: str = "System") -> None:
    events.append({"time": now_time(), "type": event_type, "agent": agent,
                   "title": title, "detail": detail})


def event_icon(event_type: str) -> str:
    icons = {"thinking": "[THINK]", "reasoning": "[REASON]", "delegating": "[DELEGATE]",
             "executing": "[EXEC]", "tool": "[TOOL]", "completed": "[DONE]",
             "failed": "[FAIL]", "info": "[INFO]", "callback": "[STEP]"}
    return icons.get(event_type, "[*]")


def render_event_timeline(events: List[Dict[str, Any]], placeholder) -> None:
    """Render a live activity timeline into a replaceable st.empty() placeholder."""
    placeholder.empty()
    with placeholder.container():
        st.markdown("#### Live Agent Activity")
        if not events:
            st.caption("No events yet.")
            return
        for event in events[-30:]:
            icon = event_icon(event.get("type", "info"))
            st.markdown(
                f"""
                <div style="padding:10px 12px;margin-bottom:8px;border-radius:10px;
                            border:1px solid rgba(128,128,128,0.25);
                            border-left:3px solid rgba(120,120,120,0.55);">
                    <span style="font-family:monospace;font-size:0.72rem;letter-spacing:0.5px;
                                 padding:1px 6px;border-radius:4px;
                                 background:rgba(128,128,128,0.15);">{icon}</span>
                    <b>&nbsp;{event.get('title','')}</b><br/>
                    <span style="font-size:0.85rem;opacity:0.75;">
                        {event.get('time','')} &middot; {event.get('agent','System')}
                    </span><br/>
                    <span>{event.get('detail','')}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# Config / Ollama / Metrics Helpers
# =========================================================

@st.cache_data
def load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return load_yaml_file(AGENTS_YAML_PATH), load_yaml_file(TASKS_YAML_PATH)


def get_ollama_models(base_url: str) -> List[str]:
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        return [m.get("name") for m in data.get("models", []) if m.get("name")]
    except Exception:  # noqa: BLE001
        return []


def build_llm(model_name: str, base_url: str, temperature: float, max_tokens: int = 3000) -> LLM:
    # max_tokens caps each generation so a small local model can't run away and
    # stall the UI; 3000 is plenty for the concise 10-section answer.
    return LLM(model=f"ollama/{model_name}", base_url=base_url,
               temperature=temperature, max_tokens=int(max_tokens))


def estimate_tokens(text: str) -> int:
    """Approx token count (~4 chars/token) for context-window visibility."""
    return max(1, len(text) // 4) if text else 0


def safe_json(obj: Any) -> Dict[str, Any]:
    try:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        if hasattr(obj, "__dict__"):
            return vars(obj)
        return {"value": str(obj)}
    except Exception:  # noqa: BLE001
        return {"value": str(obj)}


def build_chat_history(messages: List[Dict[str, Any]], max_messages: int) -> str:
    lines = []
    for msg in messages[-max_messages:]:
        role = msg.get("role", "unknown").upper()
        agent_name = msg.get("agent_name", "")
        content = msg.get("content", "")
        lines.append(f"{role} ({agent_name}): {content}" if role == "ASSISTANT" and agent_name
                     else f"{role}: {content}")
    return "\n".join(lines)


def calculate_context_metrics(chat_history: str, user_prompt: str, agents_config: Dict[str, Any],
                              tasks_config: Dict[str, Any], context_window_tokens: int) -> Dict[str, Any]:
    static_config_text = json.dumps({"agents": agents_config, "tasks": tasks_config}, ensure_ascii=False)
    chat_t = estimate_tokens(chat_history)
    prompt_t = estimate_tokens(user_prompt)
    static_t = estimate_tokens(static_config_text)
    total = chat_t + prompt_t + static_t
    return {
        "chat_history_tokens": chat_t,
        "user_prompt_tokens": prompt_t,
        "static_config_tokens": static_t,
        "estimated_total_input_tokens": total,
        "context_window_tokens": context_window_tokens,
        "estimated_context_usage_percent": round(total / max(1, context_window_tokens) * 100, 2),
    }


# =========================================================
# CrewAI Step Callback
# =========================================================

def make_step_callback(events: List[Dict[str, Any]], placeholder) -> Callable[[Any], None]:
    def step_callback(step_output: Any) -> None:
        try:
            raw = str(step_output)
            lowered = raw.lower()
            if "delegate" in lowered or "coworker" in lowered:
                etype, title = "delegating", "Delegation event detected"
            elif "tool" in lowered or "action" in lowered:
                etype, title = "tool", "Tool / action step captured"
            elif any(w in lowered for w in ("thought", "reason", "think")):
                etype, title = "reasoning", "Reasoning step captured"
            else:
                etype, title = "callback", "CrewAI step callback"
            add_event(events, etype, title, raw[:700], "CrewAI")
            render_event_timeline(events, placeholder)
        except Exception:  # noqa: BLE001
            pass
    return step_callback


# =========================================================
# Crew Assembly + Run
# =========================================================

def _assemble_crew(agents_config, tasks_config, llm, step_callback, chat_history, user_prompt,
                   mcp_buckets: Optional[Dict[str, List[Any]]]):
    from crewai import Task
    mcp_buckets = mcp_buckets or {"supervisor": [], "analyst": [], "scientist": []}

    supervisor = build_supervisor_agent(
        agents_config["supervisor_agent"], llm, mcp_buckets.get("supervisor"), step_callback)
    analyst = build_data_analyst_agent(
        agents_config["data_analyst_agent"], llm, mcp_buckets.get("analyst"), step_callback)
    scientist = build_data_scientist_agent(
        agents_config["data_scientist_agent"], llm, mcp_buckets.get("scientist"), step_callback)

    task_cfg = tasks_config["analytics_manager_task"]
    manager_task = Task(
        description=task_cfg["description"].format(chat_history=chat_history, user_prompt=user_prompt),
        expected_output=task_cfg["expected_output"],
    )
    crew = Crew(
        agents=[analyst, scientist],
        tasks=[manager_task],
        manager_agent=supervisor,
        process=Process.hierarchical,
        verbose=True,
        memory=False,
        cache=True,
    )
    return crew


def extract_crew_result_text(result: Any) -> str:
    if result is None:
        return ""
    for attr in ("raw", "final_output", "output"):
        value = getattr(result, attr, None)
        if value:
            return str(value)
    tasks_output = getattr(result, "tasks_output", None)
    if tasks_output:
        last = tasks_output[-1]
        for attr in ("raw", "summary", "output", "description"):
            value = getattr(last, attr, None)
            if value:
                return str(value)
    return str(result)


def run_delegation_crew(user_prompt, chat_history, agents_config, tasks_config, model_name,
                        base_url, temperature, use_mcp, events, placeholder, max_tokens=3000
                        ) -> Tuple[str, Dict[str, Any], str, bool]:
    """Run the hierarchical crew. Returns (final_text, usage_metrics, trace, mcp_used)."""
    add_event(events, "thinking", "Building local Ollama LLM", f"Model: ollama/{model_name}", "System")
    render_event_timeline(events, placeholder)
    llm = build_llm(model_name, base_url, temperature, max_tokens)
    step_callback = make_step_callback(events, placeholder)

    add_event(events, "executing", "Creating CrewAI agents",
              "Supervisor, Data Analyst, and Data Scientist are being initialized with tools.", "System")
    render_event_timeline(events, placeholder)

    trace_buffer = io.StringIO()
    mcp_used = False

    def _run(mcp_buckets):
        crew = _assemble_crew(agents_config, tasks_config, llm, step_callback,
                              chat_history, user_prompt, mcp_buckets)
        add_event(events, "delegating", "Starting hierarchical delegation",
                  "Supervisor may delegate to one or both specialists.", "Supervisor Agent")
        render_event_timeline(events, placeholder)
        with redirect_stdout(trace_buffer), redirect_stderr(trace_buffer):
            result = crew.kickoff()
        usage = safe_json(getattr(crew, "usage_metrics", None))
        return result, usage

    # Try MCP first (if enabled + available), else fall back to function tools only.
    if use_mcp and mcp_available():
        try:
            add_event(events, "tool", "Connecting to local MCP server",
                      "analytics_mcp_server (stdio) — exposing analytics tools.", "System")
            render_event_timeline(events, placeholder)
            with mcp_tools_by_agent() as buckets:
                result, usage = _run(buckets)
                mcp_used = True
        except Exception as mcp_err:  # noqa: BLE001
            add_event(events, "info", "MCP unavailable — using function tools only",
                      str(mcp_err)[:300], "System")
            render_event_timeline(events, placeholder)
            result, usage = _run(None)
    else:
        if use_mcp:
            add_event(events, "info", "MCP SDK not detected — using function tools only",
                      "Install `mcp` and `crewai-tools` to enable MCP.", "System")
            render_event_timeline(events, placeholder)
        result, usage = _run(None)

    add_event(events, "completed", "CrewAI run completed",
              "Supervisor returned the final response.", "Supervisor Agent")
    render_event_timeline(events, placeholder)
    return extract_crew_result_text(result), usage, trace_buffer.getvalue(), mcp_used


# =========================================================
# Session State
# =========================================================

for key, default in [
    ("messages", []), ("last_usage_metrics", {}), ("last_context_metrics", {}),
    ("last_delegation_trace", ""), ("last_events", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# =========================================================
# Load Config
# =========================================================

try:
    agents_config, tasks_config = load_configs()
except Exception as config_error:  # noqa: BLE001
    st.error(f"Config loading failed: {config_error}")
    st.stop()


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.title("Analytics Crew Settings")
    st.markdown("<hr style='margin-top:0;margin-bottom:0.8rem;border:none;"
                "border-top:2px solid rgba(128,128,128,0.35);'>", unsafe_allow_html=True)

    ollama_base_url = st.text_input("Ollama Base URL", value=DEFAULT_OLLAMA_BASE_URL)
    available_models = get_ollama_models(ollama_base_url)
    if available_models:
        idx = available_models.index(DEFAULT_MODEL_NAME) if DEFAULT_MODEL_NAME in available_models else 0
        selected_model = st.selectbox("Ollama Model", options=available_models, index=idx)
        st.success("Ollama connected")
    else:
        selected_model = st.text_input("Ollama Model Name", value=DEFAULT_MODEL_NAME,
                                        help="Exact model name from `ollama list`.")
        st.warning("Ollama not detected or no local models found")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.slider("Max Response Tokens", 1000, 8000, 3000, 500,
                           help="Caps each generation. Lower = faster but shorter answers.")
    max_context_messages = st.slider("Chat History Messages Sent", 2, 30, 6, 2)
    context_window_tokens = st.number_input("Model Context Window Tokens", 1024, 262144, 8192, 1024)

    use_mcp = st.toggle("Use local MCP server tools", value=False,
                        help="Connects agents to analytics_mcp_server. Falls back to function tools if unavailable.")
    st.caption(f"MCP SDK detected: {'yes' if mcp_available() else 'no'}")

    st.divider()
    st.subheader("Agents")
    for key, cfg in agents_config.items():
        with st.expander(cfg.get("name", key)):
            st.write(f"**Role:** {cfg.get('role')}")
            st.write(f"**Delegation:** `{cfg.get('allow_delegation')}`")
            st.caption(cfg.get("description", ""))

    st.divider()
    st.subheader("Function Tools")
    for agent_name, tools in FUNCTION_TOOL_CATALOG.items():
        with st.expander(f"{agent_name} ({len(tools)})"):
            for t in tools:
                st.write(f"• `{t}`")

    st.divider()
    st.subheader("MCP Tools")
    with st.expander(f"analytics_mcp_server ({len(MCP_TOOL_CATALOG)})"):
        for t in MCP_TOOL_CATALOG:
            st.write(f"• `{t}`")

    st.divider()
    st.subheader("Context Metrics")
    metrics = st.session_state.last_context_metrics
    if metrics:
        st.metric("Estimated Input Tokens", metrics.get("estimated_total_input_tokens", 0))
        st.metric("Context Usage", f"{metrics.get('estimated_context_usage_percent', 0)}%")
        st.progress(min(metrics.get("estimated_context_usage_percent", 0) / 100, 1.0))
        with st.expander("Full Context Metrics"):
            st.json(metrics)
    else:
        st.caption("Metrics appear after first message.")

    st.divider()
    st.subheader("CrewAI Usage Metrics")
    if st.session_state.last_usage_metrics:
        st.json(st.session_state.last_usage_metrics)
    else:
        st.caption("Usage metrics appear after first run.")

    st.divider()
    st.subheader("Delegation Trace")
    if st.session_state.last_delegation_trace:
        with st.expander("View CrewAI Verbose Trace"):
            st.text_area("CrewAI Delegation Logs", value=st.session_state.last_delegation_trace, height=300)
    else:
        st.caption("Delegation trace appears after first run.")

    st.divider()
    if st.button("Clear Chat"):
        for key in ("messages", "last_usage_metrics", "last_context_metrics",
                    "last_delegation_trace", "last_events"):
            st.session_state[key] = [] if key in ("messages", "last_events") else (
                "" if key == "last_delegation_trace" else {})
        st.rerun()


# =========================================================
# Main UI
# =========================================================

st.title("Multi-Agent Analytics Assistant")
st.caption("CrewAI Hierarchical Delegation  ·  Ollama  ·  Local MCP Server  ·  Streamlit")
st.markdown("<hr style='margin-top:0.2rem;margin-bottom:1rem;border:none;"
            "border-top:2px solid rgba(128,128,128,0.35);'>", unsafe_allow_html=True)
st.markdown(
    """
### Team Setup — CrewAI Hierarchical Delegation
- **Supervisor Agent** classifies the request, plans the work, and delegates.
- **Data Analyst Agent** — profiling, KPIs, dashboards, SQL, and data quality.
- **Data Scientist Agent** — ML use cases, features, evaluation, and pipelines.
- Agents use **local function tools** and, when available, **MCP server tools**.

Try: *Analyze the events_sample.csv file. Profile it, find data quality issues, suggest dashboard KPIs, and recommend ML use cases.*
"""
)
st.info(f"Running with Ollama model `{selected_model}` at `{ollama_base_url}` · "
        f"MCP: {'on' if use_mcp else 'off'}")

for message in st.session_state.messages:
    role = message.get("role")
    if role == "user":
        with st.chat_message("user"):
            st.markdown(message.get("content"))
    elif role == "assistant":
        with st.chat_message("assistant"):
            if message.get("agent_name"):
                st.markdown(f"**{message['agent_name']}**")
            st.markdown(message.get("content"))


user_prompt = st.chat_input("Ask your analytics crew...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    chat_history = build_chat_history(st.session_state.messages, max_context_messages)
    context_metrics = calculate_context_metrics(
        chat_history, user_prompt, agents_config, tasks_config, context_window_tokens)
    st.session_state.last_context_metrics = context_metrics

    run_events: List[Dict[str, Any]] = []
    add_event(run_events, "thinking", "User message received",
              "Preparing chat history and context metrics.", "Streamlit UI")
    add_event(run_events, "reasoning", "Context window estimated",
              f"~{context_metrics['estimated_total_input_tokens']} tokens "
              f"({context_metrics['estimated_context_usage_percent']}% of window).", "Streamlit UI")

    with st.chat_message("assistant"):
        st.markdown("**Supervisor Agent**")
        timeline_placeholder = st.empty()
        render_event_timeline(run_events, timeline_placeholder)

        assistant_response: Optional[str] = None
        assistant_error: Optional[str] = None

        with st.status("Running CrewAI hierarchical delegation...", expanded=True) as status:
            st.write("Supervisor is preparing and classifying the task.")
            st.write("Supervisor may delegate to the Analyst and/or Scientist.")
            st.write("Agents can call function tools and MCP tools.")
            st.write(f"Estimated input tokens: {context_metrics['estimated_total_input_tokens']} "
                     f"({context_metrics['estimated_context_usage_percent']}%)")
            try:
                start = time.time()
                response, usage_metrics, trace, mcp_used = run_delegation_crew(
                    user_prompt=user_prompt, chat_history=chat_history,
                    agents_config=agents_config, tasks_config=tasks_config,
                    model_name=selected_model, base_url=ollama_base_url,
                    temperature=temperature, use_mcp=use_mcp, max_tokens=max_tokens,
                    events=run_events, placeholder=timeline_placeholder,
                )
                elapsed = round(time.time() - start, 2)
                add_event(run_events, "completed", "Response ready",
                          f"Execution time: {elapsed}s · MCP used: {mcp_used}", "Streamlit UI")
                render_event_timeline(run_events, timeline_placeholder)

                st.session_state.last_usage_metrics = usage_metrics
                st.session_state.last_delegation_trace = trace
                st.session_state.last_events = run_events
                assistant_response = response
                status.update(label="CrewAI delegation completed", state="complete", expanded=False)
            except Exception as e:  # noqa: BLE001
                add_event(run_events, "failed", "CrewAI delegation failed", str(e)[:400], "System")
                render_event_timeline(run_events, timeline_placeholder)
                st.session_state.last_events = run_events
                status.update(label="CrewAI delegation failed", state="error", expanded=True)
                assistant_error = (
                    "CrewAI hierarchical delegation failed.\n\n"
                    f"```text\n{str(e)}\n```\n\n"
                    "Checklist:\n\n```bash\nollama serve\nollama list\n"
                    f"ollama pull {selected_model}\n```\n\n"
                    "Confirm `config/agents.yaml` and `config/tasks.yaml` exist. "
                    "If Ollama is slow, lower `max_iter` in `config/agents.yaml`."
                )

        # Final answer rendered OUTSIDE st.status so it stays visible when collapsed.
        if assistant_response:
            st.markdown("---")
            st.markdown("### Final Response")
            st.markdown(assistant_response)
            st.session_state.messages.append(
                {"role": "assistant", "agent_name": "Supervisor Agent", "content": assistant_response})
        elif assistant_error:
            st.error(assistant_error)
            st.session_state.messages.append(
                {"role": "assistant", "agent_name": "System", "content": assistant_error})
