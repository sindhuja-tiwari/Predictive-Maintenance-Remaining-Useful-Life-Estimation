# Lightweight container for edge inference.
# Small base image + a ~200KB model = deployable on a gateway box near the line.
FROM python:3.12-slim

# OpenMP runtime required by XGBoost (libgomp).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY service/ ./service/
# models/ is regenerated at build time below (see train step); no COPY needed

# Ensure a trained model exists in the image. If models/ was committed, this
# regenerates deterministically; if not, it trains from synthetic data at build.
RUN cd src && python make_synthetic_data.py && python train.py && cd ..

EXPOSE 8000
# Render provides $PORT at runtime; app.py reads it (falls back to 8000 locally).
CMD ["python", "service/app.py"]