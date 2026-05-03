"""
rag.py — Core RAG pipeline for Novus Bank knowledge base.

Week 2 additions:
  - hybrid_retrieve():  dense pgvector + BM25 fused with RRF (A2.3)
  - LiteLLM fallback:   model-agnostic completion with auto-fallback (A3.2)

Pipeline: embed_query → hybrid_retrieve (or retrieve) → assemble_context → generate

Every function is wrapped with @observe so LangFuse captures a full trace
with latency and token counts for each step.

Usage (interactive):
    python scripts/rag.py

Usage (import):
    from scripts.rag import ask
    result = ask("What is the minimum SIP amount?")
    result = ask("...", use_hybrid=True)   # use hybrid retrieval
"""

import os
import sys
import time
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# LiteLLM (A3.2 stretch) — model-agnostic completion with fallback
# ---------------------------------------------------------------------------

try:
    import litellm
    LITELLM_ENABLED = True
    litellm.set_verbose = False  # suppress request logs
except ImportError:
    LITELLM_ENABLED = False

# ---------------------------------------------------------------------------
# OpenAI client (used for embeddings; also for LLM if LiteLLM not installed)
# ---------------------------------------------------------------------------

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# LangFuse (optional tracing)
# ---------------------------------------------------------------------------

try:
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_ENABLED = True
except ImportError:
    LANGFUSE_ENABLED = False
    def observe(func=None, **kwargs):
        if func is not None:
            return func
        def decorator(f):
            return f
        return decorator
    class langfuse_context:
        @staticmethod
        def get_current_trace_id():
            return None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOP_K = 5
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "gpt-3.5-turbo"   # LiteLLM fallback if primary fails
TEMPERATURE = 0.1

SYSTEM_PROMPT = """You are Novus Assist, the AI knowledge assistant for Novus Bank.

Answer customer questions using ONLY the context passages provided below.
If the answer is not in the context, say: "I don't have that information in the
Novus Bank knowledge base. Please contact support at 1800-NOVUS for assistance."

Rules:
- Be concise and specific; cite exact figures (amounts, timelines, percentages) from context.
- Do not speculate or add information beyond what the context states.
- If context contains partial information, share what you know and acknowledge the gap.
- For urgent issues (fraud, account compromise), always include: "Call 1800-NOVUS immediately."
"""


def get_conn():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", 5433)),
        user=os.getenv("PG_USER", "novus"),
        password=os.getenv("PG_PASSWORD", "novus123"),
        dbname=os.getenv("PG_DATABASE", "novus_kb"),
    )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

@observe()
def embed_query(query: str) -> list[float]:
    """Embed a single query string using text-embedding-3-small."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=[query],
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Retrieval — dense only (Week 1 baseline)
# ---------------------------------------------------------------------------

@observe()
def retrieve(query_embedding: list[float], top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Cosine similarity search over pgvector chunks table.

    Returns top_k chunks sorted by similarity descending.
    Each result dict: {id, doc_id, chunk_index, content, similarity}
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT doc_id, chunk_index, content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "id": f"{row[0]}:{row[1]}",
            "doc_id": row[0],
            "chunk_index": row[1],
            "content": row[2],
            "similarity": float(row[3]),
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Retrieval — hybrid: dense + BM25 fused via RRF (A2.3)
# ---------------------------------------------------------------------------

@observe()
def hybrid_retrieve(query: str, query_embedding: list[float], top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Hybrid retrieval: dense cosine search + BM25 keyword search, merged via RRF.

    Step 1: Dense retrieval — top-k*2 chunks from pgvector.
    Step 2: BM25 search — all chunks loaded from DB, scored by term overlap.
    Step 3: RRF fusion — chunks ranked by 1/(k+rank_dense) + 1/(k+rank_bm25).
    Step 4: Return top_k results.

    Why 2x for dense: RRF needs a wider candidate pool from the dense side so
    that documents present in BM25 but ranked lower in dense can still surface.
    """
    from scripts.bm25_scratch import bm25_simple, load_chunks_from_db
    from scripts.rrf_scratch import reciprocal_rank_fusion

    # Dense candidates (2x pool for RRF)
    dense_results = retrieve(query_embedding, top_k=top_k * 2)

    # BM25 over all chunks (offline — no extra embedding cost)
    all_chunks = load_chunks_from_db()
    bm25_results = bm25_simple(all_chunks, query)[: top_k * 2]

    if not bm25_results:
        # If BM25 finds nothing, fall back to dense-only
        return dense_results[:top_k]

    fused = reciprocal_rank_fusion(dense_results, bm25_results, k=60)
    return fused[:top_k]


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

@observe()
def assemble_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        sim = f"similarity: {chunk['similarity']:.3f}" if "similarity" in chunk else "keyword match"
        parts.append(
            f"[{i}] Source: {chunk['doc_id']} ({sim})\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Generation — LiteLLM with fallback (A3.2) or raw OpenAI
# ---------------------------------------------------------------------------

@observe()
def generate(query: str, context: str) -> str:
    """Call the LLM with the assembled context to produce a grounded answer.

    Uses LiteLLM if installed (enables model-agnostic fallback).
    Falls back to raw OpenAI SDK if LiteLLM is not available.

    LiteLLM benefit: if gpt-4o-mini fails (quota, outage, bad model name),
    it automatically retries with FALLBACK_MODEL. The caller sees a response
    either way — no custom retry logic needed.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    if LITELLM_ENABLED:
        response = litellm.completion(
            model=CHAT_MODEL,
            fallbacks=[FALLBACK_MODEL],
            messages=messages,
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    else:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=TEMPERATURE,
            messages=messages,
        )
        return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@observe()
def ask(query: str, top_k: int = TOP_K, use_hybrid: bool = False) -> dict[str, Any]:
    """End-to-end RAG call. Returns a result dict suitable for eval harness.

    Args:
        query:       Customer question.
        top_k:       Number of chunks to retrieve.
        use_hybrid:  If True, use hybrid_retrieve (BM25 + dense + RRF).
                     If False, use dense-only retrieve (Week 1 baseline).

    Result keys:
        query            — original question
        answer           — LLM-generated answer
        retrieved_chunks — list of chunk dicts
        context          — assembled context string
        retrieval_mode   — "hybrid" or "dense"
        trace_id         — LangFuse trace ID (None if not configured)
        elapsed_seconds  — wall-clock time for the full pipeline
    """
    t0 = time.time()

    query_vec = embed_query(query)

    if use_hybrid:
        chunks = hybrid_retrieve(query, query_vec, top_k=top_k)
        retrieval_mode = "hybrid"
    else:
        chunks = retrieve(query_vec, top_k=top_k)
        retrieval_mode = "dense"

    context = assemble_context(chunks)
    answer = generate(query, context)

    trace_id = langfuse_context.get_current_trace_id() if LANGFUSE_ENABLED else None

    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": chunks,
        "context": context,
        "retrieval_mode": retrieval_mode,
        "trace_id": trace_id,
        "elapsed_seconds": round(time.time() - t0, 2),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Novus Bank RAG — interactive mode")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid retrieval (BM25 + dense + RRF)")
    args = parser.parse_args()

    mode = "HYBRID" if args.hybrid else "DENSE"
    print(f"Novus Bank RAG — interactive mode [{mode}]. Type 'quit' to exit.\n")
    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in {"quit", "exit"}:
            break
        result = ask(query, use_hybrid=args.hybrid)
        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: {[c['doc_id'] for c in result['retrieved_chunks']]}")
        print(f"Mode: {result['retrieval_mode']}  |  Elapsed: {result['elapsed_seconds']}s\n")
