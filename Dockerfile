# Multi-Agent Analytics Assistant — Streamlit app image.
# Ollama runs as a separate service (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps kept minimal; wheels cover pandas/duckdb/scikit-learn/scipy.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# In containers the app talks to the 'ollama' compose service, not localhost.
ENV OLLAMA_BASE_URL=http://ollama:11434 \
    OLLAMA_MODEL=llama3.2:3b

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
