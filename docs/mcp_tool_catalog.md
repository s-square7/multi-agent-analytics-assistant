# MCP Tool Catalog — analytics_mcp_server

The local MCP server (`mcp_server/server.py`) exposes ten reusable analytics tools
over stdio. Agents connect via `crewai_tools.MCPServerAdapter`.

| # | Tool | Purpose | Used by |
|---|------|---------|---------|
| 1 | `mcp_profile_csv` | Rows, columns, dtypes, missing values, duplicates, sample rows | Analyst, Scientist |
| 2 | `mcp_run_duckdb_query` | Run read-only SQL against a sample CSV via DuckDB (CSV exposed as table `data`) | Analyst |
| 3 | `mcp_validate_sql` | Read-only + hygiene validation (LIMIT, no `SELECT *`, date filter) | Supervisor, Analyst |
| 4 | `mcp_detect_data_quality_issues` | Missing, duplicate, constant, high-cardinality, negatives, IQR outliers | Analyst, Scientist |
| 5 | `mcp_generate_kpi_catalog` | KPI name, formula, grain, business use from domain + columns | Analyst, Supervisor |
| 6 | `mcp_recommend_ml_use_cases` | ML use cases with problem type, required columns, business value | Scientist, Supervisor |
| 7 | `mcp_feature_engineering_suggestions` | Engineered feature ideas for event / customer / transaction data | Scientist |
| 8 | `mcp_anomaly_detection_summary` | Anomalies via z-score, IQR, or Isolation Forest | Scientist, Analyst |
| 9 | `mcp_create_data_dictionary` | Column name, type, likely meaning, sample values | Analyst, Supervisor |
| 10 | `mcp_generate_report_markdown` | Assemble tool outputs into a final markdown report | Supervisor |

## Safety
All file-reading tools are sandboxed to `mcp_server/sample_data` (see
`mcp_server/tools/_safety.py`). SQL tools reject any non-`SELECT`/`WITH` statement
and block `DELETE, UPDATE, DROP, ALTER, INSERT, MERGE, TRUNCATE, CREATE` and
stacked statements.

## Run standalone
```bash
python mcp_server/server.py   # stdio transport
```
