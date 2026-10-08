"""HTTP API.  Run:  uvicorn grounded.api:app --reload   then open http://localhost:8000/docs"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import Agent
from .rag import AnswerEngine

app = FastAPI(title="grounded", description="A small answer engine: RAG + agents + evals.")


@lru_cache(maxsize=1)
def engine() -> AnswerEngine:
    return AnswerEngine()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    user_id: str | None = None
    k: int | None = Field(default=None, ge=1, le=10)
    mode: str | None = Field(default=None, pattern="^(bm25|vector|hybrid)$")


class AgentRequest(BaseModel):
    task: str = Field(min_length=1, max_length=4000)
    user_id: str | None = None


@app.get("/health")
def health() -> dict:
    e = engine()
    return {"status": "ok", "model": e.llm.name, "embedder": e.embedder.name,
            "chunks": len(e.retriever.chunks)}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return asdict(engine().ask(req.question, req.user_id, req.k, req.mode))


@app.post("/agent")
def run_agent(req: AgentRequest) -> dict:
    return asdict(Agent(engine()).run(req.task, req.user_id))


@app.post("/ingest")
def ingest() -> dict:
    r = engine().build_index()
    return {"chunks": len(r.chunks)}


@app.get("/users/{user_id}/memory")
def get_memory(user_id: str) -> dict:
    return asdict(engine().memory.get(user_id))


@app.delete("/users/{user_id}/memory")
def delete_memory(user_id: str) -> dict:
    e = engine()
    if not e.memory.get(user_id).preferences and not e.memory.get(user_id).topic_counts:
        raise HTTPException(404, "no memory stored for this user")
    e.memory.forget(user_id)
    return {"deleted": user_id}
