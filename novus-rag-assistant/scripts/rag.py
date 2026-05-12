"""
rag.py — Core RAG pipeline for Novus Bank knowledge base.

Week 2 additions:
  - hybrid_retrieve():  dense pgvector + BM25 fused with RRF (A2.3)
  - LiteLLM fallback:   model-agnostic completion with auto-fallback (A3.2)

Week 3 additions:
  - SemanticCache:      in-memory cosine cache, checked before any DB/LLM call (P1A.1)
  - Model router:       LLM difficulty classifier selects gpt-4o vs gpt-4o-mini (P1B.2)

Week 4 additions:
  - generate_with_confidence(): instructor-backed structured output with Confidence enum (O2.2)
  - use_confidence flag:        when True, LOW-confidence answers return HANDOFF_MESSAGE + SupportTicket
  - use_anonymizer flag:        when True, PiiAnonymizer strips PII before embed/retrieve/generate (P3.2)
  - SupportTicket wiring:       LOW confidence → generate_ticket() called; result dict gains "ticket" key (S4.1)

Pipeline: embed_query → cache_check → retrieve → assemble_context → generate(routed model)

Every function is wrapped with @observe so LangFuse captures a full trace
with latency and token counts for each step.

Usage (interactive):
    python scripts/rag.py

Usage (import):
    from scripts.rag import ask
    result = ask("What is the minimum SIP amount?")
    result = ask("...", use_hybrid=True)       # hybrid retrieval
    result = ask("...", use_cache=True)        # enable semantic cache
    result = ask("...", use_router=True)       # enable model router
    result = ask("...", use_confidence=True)   # confidence-gated answers (O2.2)
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

# ---------------------------------------------------------------------------
# Week 3 — Semantic cache (module-level singleton, opt-in via use_cache=True)
# ---------------------------------------------------------------------------

from scripts.semantic_cache import SemanticCache
_cache = SemanticCache(threshold=0.92)


# ---------------------------------------------------------------------------
# Week 3 — Model router (opt-in via use_router=True)
# ---------------------------------------------------------------------------

from scripts.difficulty_classifier import route_model_llm as _route_model_llm

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
# Week 4 — PII anonymizer (P3.2, opt-in)
# ---------------------------------------------------------------------------

try:
    from scripts.pii_anonymizer import PiiAnonymizer, redaction_audit_log
    PII_ANONYMIZER_AVAILABLE = True
except ImportError:
    PII_ANONYMIZER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Week 4 — Instructor client for structured outputs (O2.2)
# ---------------------------------------------------------------------------

try:
    import instructor
    from scripts.response_models import (
        AnswerWithConfidence, Confidence, HANDOFF_MESSAGE,
        SupportTicket, generate_ticket,
    )
    _instructor_client = instructor.from_openai(client)
    INSTRUCTOR_ENABLED = True
except ImportError:
    INSTRUCTOR_ENABLED = False

CONFIDENCE_SYSTEM_PROMPT = """You are Novus Assist, the AI knowledge assistant for Novus Bank.

Answer the customer's question using ONLY the context passages provided.
Assign a confidence level based strictly on how well the context supports the answer:
  HIGH   — every factual claim is explicitly stated in the context.
  MEDIUM — most claims are in context; minor inference or paraphrase required.
  LOW    — the context only tangentially relates, or the question is outside the knowledge base.

Rules:
- Be concise and cite exact figures from context.
- Do not speculate beyond what context states.
- For urgent issues (fraud, account compromise), always include: "Call 1800-NOVUS immediately."
"""


@observe()
def generate_with_confidence(
    query: str, context: str, model: str | None = None
) -> tuple[str, str, str]:
    """Generate an answer with a structured confidence score via instructor.

    Returns:
        (answer, confidence, reasoning)
        If confidence == LOW, answer is replaced with HANDOFF_MESSAGE.
        Falls back to plain generate() if instructor is not installed.
    """
    if not INSTRUCTOR_ENABLED:
        return generate(query, context, model=model), "unknown", "instructor not installed"

    selected_model = model or CHAT_MODEL
    try:
        response: AnswerWithConfidence = _instructor_client.chat.completions.create(
            model=selected_model,
            temperature=TEMPERATURE,
            max_tokens=400,
            response_model=AnswerWithConfidence,
            messages=[
                {"role": "system", "content": CONFIDENCE_SYSTEM_PROMPT},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        )
    except Exception:
        # Fail open: if structured call fails, fall back to plain generation
        return generate(query, context, model=model), "unknown", "instructor call failed"

    answer = (
        HANDOFF_MESSAGE
        if response.confidence == Confidence.LOW
        else response.answer
    )
    return answer, response.confidence.value, response.reasoning


# ---------------------------------------------------------------------------
# Generation — LiteLLM with fallback (A3.2) or raw OpenAI
# ---------------------------------------------------------------------------

@observe()
def generate(query: str, context: str, model: str | None = None) -> str:
    """Call the LLM with the assembled context to produce a grounded answer.

    Args:
        model: Override the chat model (e.g. "gpt-4o" for complex queries via
               the model router). Defaults to CHAT_MODEL if None.

    Uses LiteLLM if installed (enables model-agnostic fallback).
    Falls back to raw OpenAI SDK if LiteLLM is not available.
    """
    selected_model = model or CHAT_MODEL
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    if LITELLM_ENABLED:
        response = litellm.completion(
            model=selected_model,
            fallbacks=[FALLBACK_MODEL],
            messages=messages,
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    else:
        response = client.chat.completions.create(
            model=selected_model,
            temperature=TEMPERATURE,
            messages=messages,
        )
        return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@observe()
def ask(
    query: str,
    top_k: int = TOP_K,
    use_hybrid: bool = False,
    use_cache: bool = False,
    use_router: bool = False,
    use_confidence: bool = False,
    use_anonymizer: bool = False,
) -> dict[str, Any]:
    """End-to-end RAG call. Returns a result dict suitable for eval harness.

    Args:
        query:          Customer question.
        top_k:          Number of chunks to retrieve.
        use_hybrid:     If True, use hybrid_retrieve (BM25 + dense + RRF).
        use_cache:      If True, check SemanticCache before embedding/retrieving.
                        Cache is a module-level singleton (_cache). Eval harness
                        leaves this False so results stay deterministic.
        use_router:     If True, run LLM difficulty classifier to select gpt-4o
                        (complex) vs gpt-4o-mini (simple) for generation.
        use_confidence: If True, use generate_with_confidence() — returns a
                        confidence score and replaces LOW-confidence answers with
                        HANDOFF_MESSAGE. Defaults False so eval harness is unaffected.

    Result keys:
        query               — original question
        answer              — LLM-generated answer (or HANDOFF_MESSAGE if LOW)
        retrieved_chunks    — list of chunk dicts (empty on cache hit)
        context             — assembled context string (empty on cache hit)
        retrieval_mode      — "hybrid", "dense", or "cache_hit"
        model_used          — which chat model generated the answer
        cache_similarity    — cosine sim of cache hit (None if miss)
        confidence           — "high"/"medium"/"low"/"unknown" (None if use_confidence=False)
        confidence_reasoning — one-sentence reason for confidence level (None if disabled)
        pii_redacted         — True if PII was detected and anonymized (None if disabled)
        trace_id             — LangFuse trace ID (None if not configured)
        elapsed_seconds      — wall-clock time for the full pipeline
    """
    t0 = time.time()

    # --- PII anonymization (P3.2): anonymize before ANY LLM call ---
    anonymizer  = None
    clean_query = query
    if use_anonymizer and PII_ANONYMIZER_AVAILABLE:
        anonymizer  = PiiAnonymizer()
        clean_query = anonymizer.anonymize(query)

    query_vec = embed_query(clean_query)

    # --- Semantic cache check (P1A.1) ---
    if use_cache:
        hit = _cache.get(query, query_vec)
        if hit:
            trace_id = langfuse_context.get_current_trace_id() if LANGFUSE_ENABLED else None
            return {
                "query":                query,
                "answer":               hit["answer"],
                "retrieved_chunks":     [],
                "context":              "",
                "retrieval_mode":       "cache_hit",
                "model_used":           "cache",
                "cache_similarity":     hit["cache_similarity"],
                "difficulty_score":     None,
                "confidence":           None,
                "confidence_reasoning": None,
                "ticket":               None,
                "pii_redacted":         None,
                "trace_id":             trace_id,
                "elapsed_seconds":      round(time.time() - t0, 2),
            }

    # --- Retrieval ---
    if use_hybrid:
        chunks = hybrid_retrieve(query, query_vec, top_k=top_k)
        retrieval_mode = "hybrid"
    else:
        chunks = retrieve(query_vec, top_k=top_k)
        retrieval_mode = "dense"

    context = assemble_context(chunks)

    # --- Model routing (P1B.2) ---
    if use_router:
        routed_model, difficulty_score = _route_model_llm(query)
    else:
        routed_model, difficulty_score = CHAT_MODEL, None

    # --- Generation (plain or confidence-gated) — uses clean_query ---
    if use_confidence:
        raw_answer, confidence, confidence_reasoning = generate_with_confidence(
            clean_query, context, model=routed_model
        )
    else:
        raw_answer = generate(clean_query, context, model=routed_model)
        confidence = None
        confidence_reasoning = None

    # --- Restore PII in the answer ---
    answer = anonymizer.restore(raw_answer) if anonymizer else raw_answer
    pii_redacted = bool(anonymizer and anonymizer.has_pii()) if use_anonymizer else None

    # --- P3.3: Audit log — only when PII was actually found ---
    if anonymizer and pii_redacted:
        try:
            redaction_audit_log(
                query=query,
                anonymizer=anonymizer,
                intent="rag_query",   # rag.py has no intent classifier; use fixed label
                trace_id=str(langfuse_context.get_current_trace_id()) if LANGFUSE_ENABLED else None,
            )
        except Exception:
            pass  # audit log failure must never break the pipeline

    # --- S4.1: generate support ticket on LOW confidence (use_confidence=True path only) ---
    # Ticket is generated on the RESTORED query so human agents see original PII.
    ticket = None
    if use_confidence and confidence == "low" and INSTRUCTOR_ENABLED:
        try:
            ticket = generate_ticket(query, context)
        except Exception:
            ticket = None  # never let ticket generation break the main pipeline

    # --- Store in cache for future hits ---
    if use_cache:
        _cache.set(clean_query, query_vec, answer)

    trace_id = langfuse_context.get_current_trace_id() if LANGFUSE_ENABLED else None

    return {
        "query":                query,         # original — do NOT log if pii_redacted=True
        "answer":               answer,
        "retrieved_chunks":     chunks,
        "context":              context,
        "retrieval_mode":       retrieval_mode,
        "model_used":           routed_model,
        "cache_similarity":     None,
        "difficulty_score":     difficulty_score,
        "confidence":           confidence,
        "confidence_reasoning": confidence_reasoning,
        "ticket":               ticket,        # SupportTicket | None (S4.1, only on LOW)
        "pii_redacted":         pii_redacted,
        "trace_id":             trace_id,
        "elapsed_seconds":      round(time.time() - t0, 2),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Novus Bank RAG — interactive mode")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid retrieval (BM25 + dense + RRF)")
    parser.add_argument("--cache",  action="store_true", help="Enable semantic cache (P1A)")
    parser.add_argument("--router", action="store_true", help="Enable LLM model router (P1B)")
    args = parser.parse_args()

    flags = []
    if args.hybrid: flags.append("HYBRID")
    if args.cache:  flags.append("CACHE")
    if args.router: flags.append("ROUTER")
    mode = "+".join(flags) if flags else "DENSE"
    print(f"Novus Bank RAG — interactive mode [{mode}]. Type 'quit' to exit.\n")
    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in {"quit", "exit"}:
            break
        result = ask(query, use_hybrid=args.hybrid, use_cache=args.cache, use_router=args.router)
        print(f"\nAnswer: {result['answer']}")
        if result["retrieval_mode"] == "cache_hit":
            print(f"[CACHE HIT] similarity={result['cache_similarity']}")
        else:
            print(f"Sources: {[c['doc_id'] for c in result['retrieved_chunks']]}")
        print(f"Mode: {result['retrieval_mode']}  |  Model: {result['model_used']}  |  Elapsed: {result['elapsed_seconds']}s\n")
