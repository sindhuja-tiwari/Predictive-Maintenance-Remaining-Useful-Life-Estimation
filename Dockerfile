# Lightweight container for edge inference: small base image + a ~1.5 MB model,
# deployable on a gateway box near the line.
FROM python:3.12-slim

# OpenMP runtime required by XGBoost (libgomp).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ship the committed model trained on real FD001 (retrain with src/train.py).
# data/ is needed only for the dashboard's demo engines.
COPY src/ ./src/
COPY service/ ./service/
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 8000
# Render provides $PORT at runtime; falls back to 8000 locally.
CMD ["sh", "-c", "exec gunicorn --chdir service --bind 0.0.0.0:${PORT:-8000} --workers 2 --timeout 60 app:app"]
