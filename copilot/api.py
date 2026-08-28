"""
FastAPI endpoint for the Maintenance Copilot.

Endpoints:
  GET  /health         liveness + which subsystems are active (LLM, KG)
  POST /ask            {"question": "..."} -> {"answer": "...", "trace": [...]}
  POST /predict_rul    {"cycles": [...], "unit_id": 1} -> RUL result (direct tool access)
  POST /reindex        rebuild the RAG index after adding manuals

Run:  uvicorn copilot.api:app --host 0.0.0.0 --port 8100
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel

from . import agent as agent_mod
from . import rag, rul_tool, graph_kg

app = FastAPI(title="Turbofan Maintenance Copilot")


class AskBody(BaseModel):
    question: str


class RulBody(BaseModel):
    cycles: list
    unit_id: int = 1


@app.on_event("startup")
def _startup():
    # Ensure the RAG index exists (builds on first call otherwise).
    try:
        rag.build_index()
    except Exception as e:
        print(f"[api] index build deferred: {e}")


@app.get("/health")
def health():
    provider = ("anthropic" if os.environ.get("ANTHROPIC_API_KEY")
                else "openai" if os.environ.get("OPENAI_API_KEY") else None)
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
    return rul_tool.predict_rul(body.cycles, unit_id=body.unit_id)


@app.post("/reindex")
def reindex():
    n = rag.build_index(force=True)
    return {"reindexed_chunks": n}