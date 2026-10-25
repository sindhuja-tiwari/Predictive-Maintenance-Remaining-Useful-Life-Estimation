# Maintenance Copilot API (FastAPI + LangGraph + Chroma). Build from the repo root:
#   docker build -f copilot/Dockerfile -t rul-copilot .
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY copilot/requirements.txt copilot/requirements.txt
RUN pip install --no-cache-dir -r copilot/requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY copilot/ ./copilot/

EXPOSE 8100
CMD ["uvicorn", "copilot.api:app", "--host", "0.0.0.0", "--port", "8100"]
