# Architecture — Multi-Agent Analytics Assistant

## Overview
A CrewAI **hierarchical** crew answers analytics questions. A Supervisor Agent
(the manager) classifies each request, plans the work, delegates to one or both
specialists, then merges their output into one structured final answer. Each agent
has **local function tools**; specialists additionally use tools from a **local MCP
server** when it is available. The whole thing runs locally against **Ollama**, with
a **Streamlit** chat UI that shows a live activity timeline and context/usage metrics.

## Component diagram
```
                         ┌────────────────────────────────┐
                         │        Streamlit app.py         │
                         │  chat · timeline · metrics · UI │
                         └───────────────┬────────────────┘
                                         │ user_prompt + chat_history
                                         ▼
                         ┌────────────────────────────────┐
                         │   Supervisor Agent (manager)    │
                         │  CrewAI Process.hierarchical    │
                         │  fn tools: classify / plan /    │
                         │  summarize / validate / context │
                         └───────┬───────────────┬─────────┘
                        delegate │               │ delegate
                                 ▼               ▼
             ┌───────────────────────┐   ┌───────────────────────┐
             │   Data Analyst Agent  │   │  Data Scientist Agent │
             │ fn: profile / kpis /  │   │ fn: problem type /    │
             │ dashboard / sql /     │   │ features / risks /    │
             │ explain               │   │ metrics / pipeline    │
             └───────────┬───────────┘   └───────────┬───────────┘
                         │  MCPServerAdapter (stdio)  │
                         └─────────────┬──────────────┘
                                       ▼
                         ┌────────────────────────────────┐
                         │  analytics_mcp_server (FastMCP) │
                         │  10 tools · pandas · duckdb ·   │
                         │  sqlglot · scikit-learn · scipy │
                         └───────────────┬────────────────┘
                                         ▼
                         ┌────────────────────────────────┐
                         │  sample_data/  (sandboxed)      │
                         │  events · transactions · custs  │
                         └────────────────────────────────┘

        LLM for every agent:  Ollama  →  ollama/llama3.2:3b  @ localhost:11434
```

## Request lifecycle
1. The user sends a message in the Streamlit chat.
2. `app.py` builds a trimmed chat history and estimates context-window usage.
3. A hierarchical `Crew` is assembled: the Supervisor is the `manager_agent`; the
   Analyst and Scientist are worker agents. If MCP is enabled and importable, an
   `MCPServerAdapter` launches `mcp_server/server.py` and its tools are distributed
   to agents per `mcp_integration.MCP_TOOL_ROUTING`.
4. `crew.kickoff()` runs. The Supervisor classifies + plans, delegates via CrewAI's
   built-in delegation tools, and collects specialist results.
5. A `step_callback` streams events to the live timeline; stdout/stderr is captured
   as a delegation trace (never shown as a raw stack trace to the user).
6. The Supervisor returns one answer in the 10-section format from `tasks.yaml`.

## Design decisions
- **Function tools vs MCP tools.** Function tools are lightweight, deterministic
  Python wrapped as CrewAI tools — always available. MCP tools are heavier, reusable
  analytics (pandas/duckdb/sklearn) served over a standard protocol, and are optional.
  If MCP can't start, the crew degrades gracefully to function tools only.
- **Logic separated from framework.** Every tool's logic is a plain function, so the
  test suite exercises it without needing CrewAI or a running MCP transport.
- **Safety by construction.** File access is sandboxed to `sample_data`; SQL is
  restricted to read-only `SELECT`/`WITH`; no shell execution; errors surface as
  clean messages, not stack traces.

## Key modules
| Path | Responsibility |
|------|----------------|
| `app.py` | Streamlit UI, crew assembly, live timeline, metrics |
| `agents/` | Agent builders (attach tools + callbacks) |
| `function_tools/` | 15 local function tools (5 per agent) |
| `mcp_server/` | FastMCP server + 10 tool implementations + sample data |
| `mcp_integration.py` | MCP adapter bridge + per-agent tool routing |
| `config/` | `agents.yaml`, `tasks.yaml` |
| `tests/` | pytest suite for function + MCP tools |
