"""
api.py — FastAPI wrapper for the Novus Bank RAG pipeline (D1.1).

Endpoints:
    GET  /        — Serves the web UI (frontend/index.html)
    GET  /health  — ALB health check → {"status": "ok"}
    POST /query   — RAG query       → {"answer": str, "trace_id": str|null, "model_used": str, ...}
    POST /ingest  — Corpus ingest   → {"status": "done", "elapsed_seconds": float}

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8080 --reload

Test:
    curl http://localhost:8080/health
    curl -X POST http://localhost:8080/query \
         -H 'Content-Type: application/json' \
         -d '{"query": "What is the return window?"}'
"""

import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from scripts.rag import ask

app = FastAPI(title="Novus Bank RAG API", version="1.0.0")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    mode: str = "dense"          # "dense" | "hybrid" | "cache" | "router"


class QueryResponse(BaseModel):
    answer: str
    trace_id: str | None
    model_used: str
    retrieval_mode: str
    cache_similarity: float | None
    elapsed_seconds: float
    context: str                    # assembled context — needed for remote faithfulness judging
    retrieved_chunks: list[dict]    # chunk dicts — needed for remote hit rate + MRR


class IngestResponse(BaseModel):
    status: str
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

_FRONTEND = Path(__file__).parent / "frontend" / "index.html"


@app.get("/")
def frontend():
    return FileResponse(_FRONTEND)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        use_hybrid = req.mode == "hybrid"
        use_cache  = req.mode == "cache"
        use_router = req.mode == "router"
        result = ask(
            req.query,
            use_hybrid=use_hybrid,
            use_cache=use_cache,
            use_router=use_router,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        answer=result["answer"],
        trace_id=result.get("trace_id"),
        model_used=result.get("model_used", "unknown"),
        retrieval_mode=result["retrieval_mode"],
        cache_similarity=result.get("cache_similarity"),
        elapsed_seconds=result["elapsed_seconds"],
        context=result.get("context", ""),
        retrieved_chunks=result.get("retrieved_chunks", []),
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    try:
        t0 = time.time()
        from scripts.setup_db import setup as run_setup  # idempotent — safe to call every time
        run_setup()
        from scripts.ingest import ingest as run_ingest
        run_ingest()
        elapsed = round(time.time() - t0, 2)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return IngestResponse(status="done", elapsed_seconds=elapsed)
