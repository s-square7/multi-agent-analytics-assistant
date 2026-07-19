# Project Brief — Requirements → Implementation Mapping

**Programme:** Summer Training and Internship Programme on Machine Learning & Agentic AI
**Institution:** Electronics & ICT Academy, Indian Institute of Technology Roorkee
**Project:** Level 2 — Multi-Agent Analytics Assistant with Function Tools and a Local MCP Server
**Difficulty:** Level 2 — Intermediate to Advanced
**Author:** Shuvam Saren

---

## 1. Business scenario

A company receives large volumes of business and event data from CSV files, JSON logs,
application events, customer transactions, and databases. Analysts currently inspect
this data by hand: writing SQL, building KPI reports, hunting for data-quality issues,
and proposing machine-learning use cases. The process is slow and does not scale.

The goal is an internal AI analytics assistant that understands a user's question,
delegates it to specialist agents, uses function tools for local reasoning and an MCP
server for reusable analytics tooling, and returns one structured answer in a chat UI —
built entirely from free and open-source components.

## 2. Agents required

| Agent | Role | Implemented in |
|---|---|---|
| Supervisor Agent | Classifies requests, plans work, delegates, validates and merges output | `agents/supervisor_agent.py` |
| Data Analyst Agent | Profiling, KPIs, SQL validation, dashboards, business insight | `agents/data_analyst_agent.py` |
| Data Scientist Agent | ML problem framing, feature engineering, data quality, metrics, pipelines | `agents/data_scientist_agent.py` |

Roles, goals, backstories, and iteration limits are declared in `config/agents.yaml`;
task templates live in `config/tasks.yaml`.

## 3. Function tools (15 total — 5 per agent)

**Supervisor** (`function_tools/supervisor_tools.py`)

| Tool | Purpose |
|---|---|
| `classify_user_request` | Classify intent as analytics / data_science / sql / dashboard / data_quality / architecture / mixed, and recommend an agent |
| `create_agent_work_plan` | Produce a step-by-step plan for the specialists |
| `summarize_chat_history` | Compress prior turns so the context window stays small |
| `validate_final_response_structure` | Check for the required sections (Direct Answer, Architecture, Tools Used, Step-by-Step Plan, Risks, Final Recommendation) |
| `estimate_context_usage` | Estimate input tokens against the context window and report usage percentage |

**Data Analyst** (`function_tools/analyst_tools.py`)

| Tool | Purpose |
|---|---|
| `profile_dataframe` | Rows, columns, dtypes, missing values, duplicates, sample records |
| `suggest_kpi_metrics` | Domain- and column-aware KPI recommendations |
| `generate_dashboard_layout` | Propose a dashboard structure for the suggested metrics |
| `validate_sql_safety` | Read-only safety validation of a SQL query |
| `explain_query_result` | Translate a metric/trend/change into stakeholder language |

**Data Scientist** (`function_tools/scientist_tools.py`)

| Tool | Purpose |
|---|---|
| `recommend_ml_problem_type` | Classification / regression / clustering / forecasting framing |
| `suggest_feature_engineering` | Feature ideas derived from the schema |
| `detect_ml_data_risks` | Missingness, outliers, leakage and consistency risks |
| `recommend_evaluation_metrics` | Metrics appropriate to the problem type and class balance |
| `create_ml_pipeline_plan` | End-to-end pipeline outline from ingestion to monitoring |

Shared helpers sit in `function_tools/_common.py`.

## 4. MCP server (10 tools)

`mcp_server/server.py` exposes `analytics_mcp_server` over stdio via FastMCP, grouped as:

- `csv_profile_tools.py` — dataset profiling
- `sql_tools.py` — safe DuckDB query execution and SQL parsing (sqlglot)
- `data_quality_tools.py` — quality checks and anomaly detection
- `kpi_tools.py` — KPI catalog construction
- `ml_tools.py` — scikit-learn / scipy modelling helpers
- `report_tools.py` — structured report assembly
- `_safety.py` — path sandboxing and read-only SQL enforcement

Full signatures and examples: [`mcp_tool_catalog.md`](mcp_tool_catalog.md).

## 5. Sample datasets

`mcp_server/sample_data/` ships `events_sample.csv`, `transactions_sample.csv`, and
`customers_sample.csv`, each seeded with deliberate quality issues (missing values,
duplicates, inconsistent types) so the data-quality tooling has something real to find.

## 6. Constraints satisfied

- Free and open-source stack only — no paid APIs.
- Local inference through Ollama (`llama3.2:3b`); no data leaves the machine.
- Streamlit chat UI with a structured final answer.
- Hierarchical CrewAI delegation rather than a flat sequential crew.
- Deterministic tools covered by a 31-test pytest suite.
