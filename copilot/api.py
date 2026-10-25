"""
FastAPI endpoint for the Maintenance Copilot.

Endpoints:
  GET  /health         liveness + which subsystems are active (LLM, KG)
  POST /ask            {"question": "..."} -> {"answer": "...", "trace": [...]}
  POST /predict_rul    {"cycles": [...], "unit_id": 1} -> RUL result (direct tool access)
  POST /reindex        rebuild the RAG index after adding manuals

Run:  uvicorn copilot.api:app --host 0.0.0.0 --port 8100
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import agent as agent_mod
from . import rag, rul_tool, graph_kg


@asynccontextmanager
async def lifespan(_app):
    # Build the RAG index up front (otherwise it is built on the first query).
    try:
        rag.build_index()
    except Exception as e:
        print(f"[api] index build deferred: {e}")
    yield


app = FastAPI(title="Turbofan Maintenance Copilot", lifespan=lifespan)


class AskBody(BaseModel):
    question: str


class RulBody(BaseModel):
    cycles: list[dict]
    unit_id: int = 1


@app.get("/health")
def health():
    provider = agent_mod.provider()
    return {
        "status": "up",
        "llm_provider": provider or "none (retrieval-only)",
        "knowledge_graph": graph_kg.available(),
        "embed_model": rag.EMBED_MODEL,
    }


@app.post("/ask")
def ask(body: AskBody):
    return agent_mod.get_agent().ask(body.question)


@app.post("/predict_rul")
def predict_rul(body: RulBody):
    out = rul_tool.predict_rul(body.cycles, unit_id=body.unit_id)
    if "error" in out:
        raise HTTPException(status_code=400, detail=out["error"])
    return out


@app.post("/reindex")
def reindex():
    n = rag.build_index(force=True)
    return {"reindexed_chunks": n}