# Pushing this project to GitHub

Suggested repository name: **`multi-agent-analytics-assistant`**

Suggested description:
> Local multi-agent analytics assistant — CrewAI hierarchical delegation, Ollama, a FastMCP server, 25 tools, and a Streamlit UI. Capstone for the ML & Agentic AI programme, E&ICT Academy, IIT Roorkee.

Suggested topics/tags:
`crewai` `agentic-ai` `multi-agent-systems` `mcp` `model-context-protocol` `ollama`
`streamlit` `llm` `data-analytics` `duckdb` `python` `iit-roorkee`

## Commands

```bash
cd multi-agent-analytics-assistant
git init
git add .
git commit -m "Multi-Agent Analytics Assistant: CrewAI + Ollama + local MCP server"
git branch -M main
git remote add origin https://github.com/<your-username>/multi-agent-analytics-assistant.git
git push -u origin main
```

## Before you push

1. Confirm `.env` is **not** tracked (`.gitignore` already excludes it). Only `.env.example` should be committed.
2. Replace `<your-username>` in `README.md`'s clone command with your GitHub handle.
3. Take 2–3 screenshots of the running Streamlit app (chat view, activity timeline, delegation trace), save them into `assets/screenshots/`, and embed them in the README under a `## Screenshots` section:
   ```markdown
   ## Screenshots
   ![Chat interface](assets/screenshots/chat.png)
   ![Delegation trace](assets/screenshots/delegation.png)
   ```
4. Optionally record a short GIF of one full query — it makes the repo far more convincing to a recruiter than any amount of prose.
5. After the first push, add the repo description and topics from the GitHub web UI (right-hand "About" panel).

## Optional polish

- Pin the repo to your GitHub profile (Profile → Customize your pins).
- Once CI runs, swap the static tests badge in `README.md` for the live one:
  ```markdown
  ![tests](https://github.com/<your-username>/multi-agent-analytics-assistant/actions/workflows/tests.yml/badge.svg)
  ```
