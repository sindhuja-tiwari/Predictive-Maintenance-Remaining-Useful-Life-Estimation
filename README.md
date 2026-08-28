# Predictive Maintenance — Remaining Useful Life (RUL) for Industrial Equipment

Estimating the **Remaining Useful Life** of turbofan engines from time-series
sensor data to move maintenance from *reactive* (fix after failure) and
*scheduled* (fix on a fixed calendar, wasting healthy component life) to
**condition-based / predictive** — servicing each unit exactly when its measured
degradation warrants it. This is the maintenance strategy Siemens frames its
industrial digitalization and MindSphere/Senseye offerings around: using sensor
telemetry and **degradation modeling** to **prevent unplanned downtime**.

## Dataset
NASA **C-MAPSS Turbofan Degradation** (subset FD001). Each engine runs from
healthy operation to failure while 21 sensors + 3 operating settings are logged
per cycle. The label is RUL = cycles remaining until failure.

> A **synthetic generator** (`src/make_synthetic_data.py`) produces data in the
> exact C-MAPSS format so the pipeline runs immediately. To use the real data,
> download FD001 from the NASA Prognostics Data Repository and drop
> `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` into `data/`.

## Approach
1. **RUL labeling** — piecewise-linear target clipped at 125 cycles (an engine is
   "as-good-as-new" until wear begins; this is the standard C-MAPSS treatment).
2. **Degradation feature engineering** — per-engine rolling mean/std (windows of
   5 and 15 cycles) to expose sensor *drift* over time, plus delta-from-initial
   readings to capture cumulative wear. Windows are grouped per unit so no signal
   leaks across engines.
3. **Model** — XGBoost regressor. Strong tabular baseline, trains in seconds,
   exports to a ~200KB artifact suited to edge inference.
4. **Evaluation** — RMSE (cycles) and the **NASA asymmetric score**, which
   penalizes *late* predictions (over-estimating RUL) more than early ones,
   reflecting the real cost of an unplanned outage.

## Run
```bash
pip install -r requirements.txt
cd src
python make_synthetic_data.py   # or drop real FD001 files into ../data
python train.py                 # -> models/rul_xgb.json + meta.json
python evaluate.py              # test-set RMSE + NASA score
```

## Edge inference service (bonus)
A single-file Flask app loads the model once and serves RUL predictions with a
tiered maintenance alert (OK / WARNING / CRITICAL). Small footprint simulates a
gateway box running inference next to the machinery instead of streaming raw
sensors to the cloud.

```bash
python service/app.py           # serves on :8000
curl localhost:8000/health
# POST recent cycles -> {"predicted_rul_cycles": 42.0, "alert": "WARNING", ...}
```

Containerized for deployment:
```bash
docker build -t rul-edge .
docker run -p 8000:8000 rul-edge
```

## Project layout
```
pdm/
├── data/                     C-MAPSS files (synthetic or real)
├── src/
│   ├── make_synthetic_data.py
│   ├── data_prep.py          load, RUL labels, feature engineering
│   ├── train.py              XGBoost training + metrics
│   └── evaluate.py           official test-split evaluation
├── models/                   trained model + metadata
├── service/app.py            Flask edge-inference API
├── Dockerfile
└── requirements.txt
```

## Extending
- Add FD002–FD004 (multiple operating conditions / fault modes).
- Swap in an LSTM/1D-CNN sequence model and compare against the XGBoost baseline.
- Add a simple dashboard plotting predicted RUL and alert history per unit.

# Maintenance Copilot

An agent that answers questions about turbofan-engine health by combining the
trained C-MAPSS RUL model with retrieval over maintenance manuals and an optional
Engine→Sensor→FailureMode knowledge graph.

## Architecture

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

- **agent node** — calls the LLM (Anthropic or OpenAI) with tool schemas; the model
  decides whether to answer or call a tool.
- **tools node** — executes requested tools and feeds results back.
- The conditional edge loops agent↔tools until the model produces a final answer.

## Tools

| Tool | Backed by | Use |
|---|---|---|
| `predict_rul` | trained XGBoost model (`../models/`) | estimate RUL + alert from sensor window |
| `search_manuals` | Chroma vector store + `all-MiniLM-L6-v2` embeddings | retrieve guidance from manuals |
| `failure_modes` | Neo4j (optional; static-map fallback) | map a sensor to failure modes + mitigations |

## Setup

```bash
pip install -r copilot/requirements.txt

# Build the RAG index over copilot/manuals/ (drop your own PDFs here first)
python -m copilot.rag

# Set an LLM key for full agentic reasoning (either provider works)
export ANTHROPIC_API_KEY=sk-ant-...       # or: export OPENAI_API_KEY=sk-...

# Run the API
uvicorn copilot.api:app --host 0.0.0.0 --port 8100
```

Without an API key, the agent runs in **retrieval-only mode**: it still answers from
the manuals via RAG, so the system is fully runnable for a demo before you add a key.

## Endpoints

```bash
# Ask the copilot
curl -X POST http://localhost:8100/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Engine 3 shows rising HPC temperature and fuel flow — what failure mode is this and what should I do?"}'

# Direct RUL tool
curl -X POST http://localhost:8100/predict_rul \
  -H "Content-Type: application/json" \
  -d '{"unit_id": 3, "cycles": [{"sensor_1": 520, "sensor_2": 640}]}'

# Rebuild index after adding manuals
curl -X POST http://localhost:8100/reindex

# Subsystem status
curl http://localhost:8100/health
```

## Optional: Neo4j knowledge graph

```bash
# Bring up copilot + Neo4j together
export ANTHROPIC_API_KEY=sk-ant-...
docker compose -f copilot/docker-compose.yml up --build

# Seed the Engine→Sensor→FailureMode graph
python -m copilot.graph_kg
```

If `NEO4J_URI` is unset or the database is unreachable, `failure_modes` falls back to
a built-in sensor→mode map, so nothing breaks.

## Adding your own manuals

Drop `.pdf`, `.md`, or `.txt` files into `copilot/manuals/` and run
`python -m copilot.rag` (or `POST /reindex`). PDFs are parsed with `pypdf`.

## Notes

- Embeddings are local (no API key, no per-call cost); only the agent's reasoning LLM
  uses an API key.
- The bundled manual is synthetic sample content — replace it with real OEM
  documentation for a production knowledge base.
