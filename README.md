# Predictive Maintenance: Remaining Useful Life (RUL) Estimation for Turbofan Engines

Estimates the **Remaining Useful Life** of turbofan engines from time-series sensor data, moving
maintenance from *reactive* (fix after failure) and *scheduled* (fix on a fixed calendar, wasting
healthy component life) to **condition-based**: service each unit when its measured degradation
warrants it. The repo has three parts: an XGBoost RUL model, a small edge-inference service, and a
LangGraph "Maintenance Copilot" that combines the model with retrieval over maintenance manuals.

## Dataset

NASA **C-MAPSS Turbofan Degradation** (subset FD001). Each engine runs from healthy operation to
failure while 21 sensors and 3 operating settings are logged per cycle. The label is RUL: the
number of cycles remaining until failure.

> **Real vs synthetic data.** `src/make_synthetic_data.py` generates data in the exact C-MAPSS
> format so the pipeline runs immediately without a download. To use the real data, download FD001
> from the NASA Prognostics Data Repository and place `train_FD001.txt`, `test_FD001.txt` and
> `RUL_FD001.txt` in `data/`. Any metrics you quote should come from the real data.

## Approach

1. **RUL labelling.** Piecewise-linear target clipped at 125 cycles (an engine is treated as
   as-good-as-new until wear begins; the standard C-MAPSS treatment).
2. **Degradation features.** Per-engine rolling mean and standard deviation (windows of 5 and 15
   cycles) to expose sensor drift, plus delta-from-initial readings to capture cumulative wear.
   Windows are grouped per unit so no signal leaks across engines.
3. **Model.** XGBoost regressor: a strong tabular baseline that trains in seconds and exports to a
   small (~200 KB) artifact suited to edge inference.
4. **Evaluation.** RMSE (in cycles) and the **NASA asymmetric score**, which penalizes late
   predictions (over-estimating RUL) more than early ones, reflecting the higher cost of an
   unplanned outage.

## Quick start (RUL model)

```bash
pip install -r requirements.txt
cd src
python make_synthetic_data.py   # skip if you placed the real FD001 files in data/
python train.py                 # trains the XGBoost model, saves it to models/
python evaluate.py              # evaluates on the official test split: RMSE and NASA score
```

## Edge inference service

A single-file Flask app loads the model once and serves RUL predictions with a tiered maintenance
alert (OK / WARNING / CRITICAL). The small footprint simulates a gateway box running inference
next to the machinery instead of streaming raw sensor data to the cloud.

```bash
python service/app.py           # serves on :8000
curl localhost:8000/health
# POST recent cycles -> {"predicted_rul_cycles": 42.0, "alert": "WARNING", ...}
```

Containerized:

```bash
docker build -t rul-edge .
docker run -p 8000:8000 rul-edge
```

## Maintenance Copilot

An agent that answers questions about engine health by combining the trained RUL model with
retrieval over maintenance manuals and an optional Engine → Sensor → FailureMode knowledge graph.

### Architecture

```
                 ┌──────────────────────────────┐
question  ─────▶ │   LangGraph state graph        │
                 │                                │
                 │   START → agent → tools →┐     │
                 │             ▲────────────┘     │
                 │             └──────────→ END   │
                 └──────┬─────────────┬───────────┘
                        │             │
           tool node dispatches to:   │
     ┌──────────────────┼─────────────┼───────────────────┐
     ▼                  ▼             ▼                    ▼
predict_rul       search_manuals   failure_modes    (LLM reasoning)
XGBoost RUL       Chroma + local   Neo4j graph
model (tool)      embeddings (RAG) (optional)
```

- The **agent node** calls the configured LLM with the tool schemas; the model decides whether to
  answer or call a tool.
- The **tools node** executes the requested tools and feeds the results back.
- A conditional edge loops agent ↔ tools until the model produces a final answer.

### Tools

| Tool             | Backed by                                           | Use                                         |
| ---------------- | --------------------------------------------------- | ------------------------------------------- |
| `predict_rul`    | trained XGBoost model (`models/`)                   | estimate RUL and alert from a sensor window |
| `search_manuals` | Chroma vector store + `all-MiniLM-L6-v2` embeddings | retrieve guidance from manuals              |
| `failure_modes`  | Neo4j (optional; static-map fallback)               | map a sensor to failure modes + mitigations |

### Setup

```bash
pip install -r copilot/requirements.txt

export ANTHROPIC_API_KEY=<your key>     # optional, see below
python -m copilot.rag                   # index the manuals in copilot/manuals/
uvicorn copilot.api:app --host 0.0.0.0 --port 8100
```

Without an API key the agent runs in **retrieval-only mode**: it still answers from the manuals
via RAG, so the system is runnable end to end before you add a key.

### Endpoints

```bash
curl -X POST http://localhost:8100/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Engine 3 shows rising HPC temperature and fuel flow. What failure mode is this and what should I do?"}'

curl -X POST http://localhost:8100/predict_rul \
  -H "Content-Type: application/json" \
  -d '{"unit_id": 3, "cycles": [{"sensor_1": 520, "sensor_2": 640}]}'

curl -X POST http://localhost:8100/reindex

curl http://localhost:8100/health
```

### Optional: Neo4j knowledge graph

```bash
export ANTHROPIC_API_KEY=<your key>
docker compose -f copilot/docker-compose.yml up --build

python -m copilot.graph_kg
```

If `NEO4J_URI` is unset or the database is unreachable, `failure_modes` falls back to a built-in
sensor → failure-mode map, so nothing breaks.

### Adding your own manuals

Drop `.pdf`, `.md` or `.txt` files into `copilot/manuals/`, then run `python -m copilot.rag` (or
`POST /reindex`). PDFs are parsed with `pypdf`.

## Project layout

```
Predictive-Maintenance-Remaining-Useful-Life-Estimation/
├── data/                     C-MAPSS files (synthetic or real)
├── src/
│   ├── make_synthetic_data.py
│   ├── data_prep.py          load, RUL labels, feature engineering
│   ├── train.py              XGBoost training + metrics
│   └── evaluate.py           official test-split evaluation
├── models/                   trained model + metadata
├── service/app.py            Flask edge-inference API
├── copilot/
│   ├── agent.py              LangGraph agent
│   ├── api.py                FastAPI endpoints
│   ├── rag.py                manual indexing + retrieval (Chroma)
│   ├── graph_kg.py           Neo4j knowledge graph
│   ├── manuals/              source documents
│   ├── requirements.txt
│   └── docker-compose.yml
├── Dockerfile                edge-inference image
└── requirements.txt          RUL model + edge service dependencies
```

## Extending

- Add FD002 to FD004 (multiple operating conditions and fault modes).
- Swap in an LSTM or 1D-CNN sequence model and compare against the XGBoost baseline.
- Add a dashboard plotting predicted RUL and alert history per unit.
