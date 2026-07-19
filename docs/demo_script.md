# Demo Script

A five-minute walkthrough of the Multi-Agent Analytics Assistant.

## 0. Prerequisites
```bash
pip install -r requirements.txt
ollama serve          # terminal 1
ollama pull llama3.2:3b
streamlit run app.py  # terminal 2 -> http://localhost:8501
```

## 1. Show the crew (30s)
Open the sidebar. Point out the three agents, the 5 function tools per agent, the
10 MCP tools under `analytics_mcp_server`, and the "Use local MCP server tools"
toggle. Note the MCP-detected indicator.

## 2. Primary demo prompt (2 min)
Paste into the chat:

> Analyze the events_sample.csv file. Profile it, find data quality issues,
> suggest dashboard KPIs, and recommend ML use cases.

While it runs, narrate the **live activity timeline**: context estimate →
LLM build → agent creation → MCP connect → hierarchical delegation → completion.

Expected final answer contains all ten sections: Direct Answer, Dataset Summary,
Data Quality Findings, Recommended KPIs, Recommended Dashboard, ML Use Cases,
Feature Engineering Ideas, Risks and Limitations, Next Steps, Agent Work Summary.

## 3. Delegation evidence (45s)
Open **Delegation Trace** in the sidebar to show the Supervisor delegating to the
Analyst and Scientist. Open **Context Metrics** and **CrewAI Usage Metrics**.

## 4. Analyst-only vs Scientist-only (1 min)
- *"Write a safe SQL query to count events per type in events_sample.csv and flag any unsafe SQL."* → Analyst path.
- *"I want to predict customer churn from the customers data — what problem type, features, metrics, and pipeline?"* → Scientist path.

## 5. Safety demo (45s)
Ask: *"Profile the file /etc/passwd."* The tools refuse — access is sandboxed to
`sample_data`. Then: *"Run DELETE FROM data."* The SQL validator blocks it.

## 6. Tests (15s)
```bash
pytest tests/ -q     # 31 passing
```
