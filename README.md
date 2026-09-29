# Predictive Maintenance: Remaining Useful Life Estimation

Estimates how many operating cycles a turbofan engine has left before failure,
using its sensor history. The goal is condition-based maintenance: servicing
each engine when its measured degradation calls for it, rather than after a
breakdown or on a fixed calendar that discards healthy component life.

Live demo: [Render URL]

## Dataset

NASA **C-MAPSS** turbofan degradation data, subset **FD001**: 100 training
engines run from healthy operation to failure, and 100 test engines whose
records stop at some point before failure. FD001 has one operating condition
and one fault mode (HPC degradation). Each cycle logs 21 sensors and 3
operating settings.

Download FD001 from the NASA Prognostics Center of Excellence Data Repository
and place `train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt` in `data/`.

`src/make_synthetic_data.py` generates data in the same format for smoke tests
only. All reported results use the real FD001 files.

## Approach

1. **RUL target.** Piecewise-linear, capped at 125 cycles. An engine is treated
   as healthy until wear begins, which is the standard treatment for C-MAPSS.
2. **Degradation features.** Per-engine rolling mean and standard deviation
   (windows of 5 and 15 cycles) to capture sensor drift, plus each sensor's
   change from its initial reading to capture cumulative wear. Features are
   computed within each engine, so no information crosses between units.
3. **Model.** XGBoost regressor. It trains in seconds and exports to a ~200 KB
   artifact small enough for edge inference.
4. **Evaluation.** The official FD001 test protocol: predict RUL at the last
   recorded cycle of each of the 100 test engines and compare against
   `RUL_FD001.txt`.

### Metrics

- **RMSE** in cycles.
- **NASA scoring function.** With `d = predicted RUL − true RUL`:

  ```
  s = Σ  exp(−d / 13) − 1    if d < 0   (early prediction)
         exp( d / 10) − 1    if d ≥ 0   (late prediction)
  ```

  Lower is better. Late predictions are penalized more steeply because an
  engine that fails before its predicted maintenance date is the costlier
  mistake. For example, a prediction 10 cycles late scores about 1.72, while
  one 10 cycles early scores about 1.16.

## Results

FD001 test set (100 engines):

| Model | RMSE (cycles) | NASA score |
|---|---|---|
| Linear regression, same features | [ ] | [ ] |
| XGBoost | [ ] | [ ] |

## Run

```bash
pip install -r requirements.txt
cd src
python train.py        # trains on data/train_FD001.txt
python evaluate.py     # official test-split evaluation
```

To try the pipeline without downloading the data, run
`python make_synthetic_data.py` first. Results from synthetic data are not
meaningful.

## Edge inference service

A single-file Flask app loads the model once and serves RUL predictions with a
tiered maintenance alert (OK / WARNING / CRITICAL). The small footprint
simulates a gateway device running inference next to the equipment instead of
streaming raw sensor data to the cloud.

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

An agent that answers questions about engine health by combining the trained
RUL model with retrieval over maintenance manuals and an optional
Engine → Sensor → FailureMode knowledge graph.

```
                    ┌──────────────────────────────┐
   question  ─────▶ │   LangGraph state graph      │
                    │                              │
                    │   START → agent → tools →┐   │
                    │             ▲────────────┘   │
                    │             └──────────→ END │
                    └──────┬─────────────┬─────────┘
                           │             │
              tool node dispatches to:   │
        ┌──────────────────┼─────────────┼──────────────────┐
        ▼                  ▼             ▼                   ▼
   predict_rul       search_manuals   failure_modes    (LLM reasoning)
   XGBoost RUL       Chroma + local   Neo4j graph
   model (tool)      embeddings (RAG) (optional)
```

- **Agent node** calls the LLM (Anthropic or OpenAI) with tool schemas, and the
  model decides whether to answer or call a tool.
- **Tools node** executes the requested tools and returns the results.
- A conditional edge loops between agent and tools until the model produces a
  final answer.

| Tool | Backed by | Use |
|---|---|---|
| `predict_rul` | Trained XGBoost model (`models/`) | Estimate RUL and alert level from a sensor window |
| `search_manuals` | Chroma vector store + `all-MiniLM-L6-v2` embeddings | Retrieve guidance from maintenance manuals |
| `failure_modes` | Neo4j (optional, with a static-map fallback) | Map a sensor to failure modes and mitigations |

### Setup

```bash
pip install -r copilot/requirements.txt
python -m copilot.rag                  # build the manual index

export ANTHROPIC_API_KEY=...           # or OPENAI_API_KEY
uvicorn copilot.api:app --host 0.0.0.0 --port 8100
```

Without an API key, the agent runs in retrieval-only mode and answers from the
manuals via RAG.

### Endpoints

```bash
curl -X POST http://localhost:8100/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Engine 3 shows rising HPC temperature and fuel flow. What failure mode is this and what should I do?"}'

# Abbreviated payload; send the full sensor set for each recent cycle.
curl -X POST http://localhost:8100/predict_rul \
  -H "Content-Type: application/json" \
  -d '{"unit_id": 3, "cycles": [{"sensor_1": 520, "sensor_2": 640}]}'

curl -X POST http://localhost:8100/reindex
curl http://localhost:8100/health
```

### Optional: Neo4j knowledge graph

```bash
docker compose -f copilot/docker-compose.yml up --build
python -m copilot.graph_kg
```

If `NEO4J_URI` is unset or the database is unreachable, `failure_modes` falls
back to a built-in sensor-to-failure-mode map.

### Adding manuals

Put `.pdf`, `.md` or `.txt` files in `copilot/manuals/`, then run
`python -m copilot.rag` or call `POST /reindex`. PDFs are parsed with `pypdf`.

## Project layout

```
Predictive-Maintenance-Remaining-Useful-Life-Estimation/
├── data/                     C-MAPSS FD001 files
├── copilot/
│   ├── manuals/              maintenance documents for retrieval
│   ├── agent.py              LangGraph agent
│   ├── api.py                FastAPI endpoints
│   ├── graph_kg.py           Neo4j knowledge graph loader
│   └── rag.py                Chroma indexing and retrieval
├── src/
│   ├── make_synthetic_data.py
│   ├── data_prep.py          loading, RUL labels, feature engineering
│   ├── train.py              XGBoost training
│   └── evaluate.py           official test-split evaluation
├── models/                   trained model and metadata
├── service/app.py            Flask edge-inference API
├── Dockerfile
└── requirements.txt
```

## Future work

- Extend to FD002–FD004 (multiple operating conditions and fault modes).
- Compare an LSTM or 1D-CNN sequence model against the XGBoost baseline.
- Add a dashboard plotting predicted RUL and alert history per engine.
