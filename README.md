# Multi-Agent Analytics Assistant

**A local, privacy-preserving multi-agent analytics chat assistant** built with CrewAI
(hierarchical delegation), Ollama (local LLM), a local **MCP server**, function tools,
and a Streamlit UI.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="CrewAI" src="https://img.shields.io/badge/CrewAI-hierarchical-FF5A5F">
  <img alt="Ollama" src="https://img.shields.io/badge/Ollama-llama3.2%3A3b-000000">
  <img alt="MCP" src="https://img.shields.io/badge/MCP-FastMCP-6E56CF">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Tests" src="https://img.shields.io/badge/tests-31%20passing-brightgreen">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-blue">
</p>

> **Capstone Project — Level 2**
> Submitted for the **Summer Training and Internship Programme on Machine Learning & Agentic AI**
> **Electronics & ICT Academy, Indian Institute of Technology Roorkee**
> Author: **Shuvam Saren** · M.Tech (Computer Science & Data Processing), IIT Kharagpur

---

## Overview

A **Supervisor Agent** classifies each request, plans the work, and delegates to a
**Data Analyst Agent** (SQL, KPIs, dashboards, data quality) and/or a **Data Scientist
Agent** (ML use cases, features, evaluation, pipelines), then returns one structured
answer. Every agent has local **function tools**; specialists also use tools from a
local **MCP server** when it is running.

Everything runs on your own machine — no API keys, no cloud inference, no data egress.

## Features

- CrewAI **hierarchical** crew: 1 manager + 2 specialists.
- **15 function tools** (5 per agent) — deterministic and unit-tested.
- **10 MCP tools** on `analytics_mcp_server` (pandas, DuckDB, sqlglot, scikit-learn, scipy).
- **Streamlit UI**: chat, live activity timeline, context + usage metrics, delegation trace.
- **Safety guardrails**: sandboxed file access, read-only SQL, no shell execution, clean errors.
- Bundled **sample datasets** with intentional data-quality issues.
- **31 passing tests** with CI on every push.

## Architecture

```
User
 └─> Streamlit Chat UI (app.py)
      └─> CrewAI Hierarchical Crew
           └─> Supervisor Agent  (classify · plan · summarize · validate · context)
                ├─> Data Analyst Agent    (profile · KPIs · dashboard · SQL · insights)
                └─> Data Scientist Agent  (problem type · features · metrics · pipeline)
                     └─> Function Tools  +  Local MCP Server (FastMCP, 10 tools)
                          └─> Ollama (llama3.2:3b) — fully local inference
```

Full component diagram, data flow, and design decisions: **[`docs/architecture.md`](docs/architecture.md)**.

## Repository structure

```
multi-agent-analytics-assistant/
├── app.py                       # Streamlit UI + crew assembly
├── mcp_integration.py           # MCP adapter bridge + per-agent tool routing
├── requirements.txt
├── Dockerfile / docker-compose.yml
├── .env.example
├── config/
│   ├── agents.yaml              # agent roles, goals, backstories, limits
│   └── tasks.yaml               # task templates
├── agents/                      # supervisor / analyst / scientist builders
├── function_tools/              # 15 local function tools (5 per agent)
├── mcp_server/
│   ├── server.py                # FastMCP server (10 tools)
│   ├── tools/                   # tool implementations + safety layer
│   └── sample_data/             # events / transactions / customers CSVs
├── tests/                       # pytest suite (31 tests)
├── docs/
│   ├── architecture.md          # system design
│   ├── mcp_tool_catalog.md      # MCP tool reference
│   ├── project_brief.md         # assignment brief → implementation mapping
│   └── demo_script.md           # walkthrough for the demo/viva
├── assets/screenshots/          # UI screenshots
└── .github/workflows/tests.yml  # CI
```

## Quickstart

**1. Clone and install**
```bash
git clone https://github.com/<your-username>/multi-agent-analytics-assistant.git
cd multi-agent-analytics-assistant
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Start Ollama and pull the model**
```bash
ollama serve            # keep running in its own terminal
ollama pull llama3.2:3b
ollama list             # confirm the model is present
```

**3. Configure (optional)**
```bash
cp .env.example .env    # adjust OLLAMA_BASE_URL / OLLAMA_MODEL if needed
```

**4. Launch**
```bash
streamlit run app.py    # not `python app.py`
```
Opens on http://localhost:8501.

### Try it

> Analyze the `events_sample.csv` file. Profile it, find data quality issues,
> suggest dashboard KPIs, and recommend ML use cases.

## Running the MCP server standalone (optional)

```bash
python mcp_server/server.py     # stdio transport
```

In the app, the **"Use local MCP server tools"** toggle connects the agents to it.
If the MCP SDK or adapter is missing, the app automatically falls back to function
tools only. Tool-by-tool reference: **[`docs/mcp_tool_catalog.md`](docs/mcp_tool_catalog.md)**.

## Tests

```bash
pytest tests/ -q     # 31 passed
```

## Deployment

The app talks to Ollama over HTTP. On your own machine that's `localhost:11434`.
When deployed, `localhost` points at the **server**, so Ollama must be reachable
from wherever the app runs. Two supported paths:

**A) Docker Compose (recommended — app + co-located Ollama):**
```bash
docker compose up --build
# one-time: pull the model into the ollama service
docker exec -it analytics_ollama ollama pull llama3.2:3b
# open http://localhost:8501
```
The app reaches Ollama at `http://ollama:11434` (the compose service name).
A GPU block is included (commented) in `docker-compose.yml`.

**B) Existing Ollama endpoint:** set the **Ollama Base URL** field in the sidebar
(or `OLLAMA_BASE_URL`) to a network-reachable Ollama server.

> Streamlit Community Cloud can't run Ollama (no local LLM server), so use a
> self-hosted VM or the Docker Compose setup for a full deployment.

## Performance & robustness

- **Max Response Tokens** slider caps each generation so the model can't stall the UI.
- **Lean delegation** — `max_iter` kept low in `config/agents.yaml`; tool caching on, memory off.
- **Cached reads** — each sample CSV is parsed once and reused across tools.
- **Forgiving tool inputs** — column-list tools also accept `"a, b"` or a JSON list,
  so a small model passing the "wrong" shape still works.
- **MCP is optional** — off by default for speed; toggle it on in the sidebar anytime.

## Safety notes

- File tools only read `mcp_server/sample_data` (path traversal is blocked).
- SQL tools allow read-only `SELECT`/`WITH` and block
  `DELETE, UPDATE, DROP, ALTER, INSERT, MERGE, TRUNCATE, CREATE` and stacked statements.
- No shell execution; failures surface as clean messages, never raw stack traces.

## Tech stack

CrewAI · Ollama (`llama3.2:3b`) · MCP (FastMCP + crewai-tools adapter) · Streamlit ·
pandas · DuckDB · sqlglot · scikit-learn · scipy · pytest · Docker.

## Acknowledgements

Developed as the Level 2 capstone for the **Summer Training and Internship Programme
on Machine Learning & Agentic AI**, conducted by the **Electronics & ICT Academy,
Indian Institute of Technology Roorkee**.

## License

Released under the [MIT License](LICENSE).
