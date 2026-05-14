"""
reranker.py — Cohere cross-encoder reranker for the Novus RAG pipeline.

Takes a list of candidate chunks (from dense or hybrid retrieval) and
re-scores them using Cohere's rerank-english-v3.0 model. The reranker
sees the full query + chunk text together, so it captures relevance that
cosine similarity misses (vocabulary mismatch, negation, numeric precision).

Usage (via rag.py):
    result = ask("...", use_hybrid=True, use_reranker=True)

Usage (standalone):
    from scripts.reranker import cohere_rerank
    reranked = cohere_rerank(query, chunks, top_n=5)

Requires:
    COHERE_API_KEY set in environment (or .env).
    `cohere` package installed (pip install cohere).

Degrades gracefully: if cohere is not installed or COHERE_API_KEY is absent,
cohere_rerank() returns the input chunks unchanged so the pipeline never breaks.
"""

import os
from typing import Any

from dotenv import load_dotenv
load_dotenv()

RERANK_MODEL = "rerank-english-v3.0"
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

try:
    import cohere as _cohere_lib
    _cohere_client = _cohere_lib.Client(COHERE_API_KEY) if COHERE_API_KEY else None
    COHERE_AVAILABLE = bool(_cohere_client)
except ImportError:
    _cohere_lib = None
    _cohere_client = None
    COHERE_AVAILABLE = False


def cohere_rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Re-rank retrieved chunks using Cohere's cross-encoder reranker.

    Args:
        query:   The original customer query.
        chunks:  List of chunk dicts from retrieve() or hybrid_retrieve().
                 Each dict must have at least a "content" key.
        top_n:   How many top chunks to return after reranking.
                 Defaults to len(chunks) — reorder everything, keep all.

    Returns:
        Chunks sorted by rerank_score descending, with "rerank_score" added
        to each dict. If Cohere is unavailable, returns chunks unchanged.
    """
    if not COHERE_AVAILABLE or not chunks:
        return chunks

    if top_n is None:
        top_n = len(chunks)

    documents = [c["content"] for c in chunks]

    try:
        response = _cohere_client.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=top_n,
        )
    except Exception:
        # Never let reranker failure break the pipeline
        return chunks

    reranked: list[dict[str, Any]] = []
    for hit in response.results:
        chunk = dict(chunks[hit.index])
        chunk["rerank_score"] = round(hit.relevance_score, 4)
        reranked.append(chunk)

    return reranked
