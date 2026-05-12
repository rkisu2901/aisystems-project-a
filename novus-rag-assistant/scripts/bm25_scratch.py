"""
bm25_scratch.py — BM25 keyword search implemented from first principles.

A2.1 deliverable: Build keyword ranking by hand before using any library.
This makes the mathematics visible — the step from term-frequency counting
to full BM25 is clear once you have this baseline.

Implementation:
  Simplified BM25 (no IDF weighting, no length normalisation — those are
  added in the full BM25 formula). Each chunk is scored by counting how many
  query tokens appear in it (term frequency, binary). Rank by that count.

Why this matters:
  Dense (embedding) retrieval excels at semantic similarity but fails for
  vocabulary mismatches — e.g. a user asks "how do I get my money back"
  but the policy document uses "refund" / "return window". BM25 doesn't
  need semantic overlap; it finds exact token matches, so it complements
  dense retrieval in the RRF fusion step (rrf_scratch.py).

Usage:
    python scripts/bm25_scratch.py
"""

from __future__ import annotations
import os, sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# A2.1 — simplified BM25 (term-frequency, no IDF)
# ---------------------------------------------------------------------------

def bm25_simple(chunks: list[dict], query: str) -> list[dict]:
    """Rank chunks by the number of query tokens they contain.

    Args:
        chunks: List of dicts with at least 'id' and 'content' keys.
        query:  Free-text query string.

    Returns:
        Ordered list of matching chunks (score > 0), highest score first.
        Chunks with zero matches are excluded.

    Why no IDF:
        IDF penalises tokens that appear in many documents. For a small
        corpus (19 docs) IDF has limited effect — common policy words like
        "account" appear everywhere. Adding IDF is the natural next step
        (see rank_bm25 library used in rag.py hybrid path).
    """
    query_tokens = set(query.lower().split())
    scored: list[tuple[int, int, dict]] = []

    for i, chunk in enumerate(chunks):
        chunk_tokens = chunk["content"].lower().split()
        score = sum(1 for t in chunk_tokens if t in query_tokens)
        scored.append((score, i, chunk))

    return [c for score, _, c in sorted(scored, key=lambda x: x[0], reverse=True) if score > 0]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def load_chunks_from_db() -> list[dict]:
    """Load all chunks from pgvector for offline BM25 testing."""
    from dotenv import load_dotenv
    import psycopg2

    load_dotenv()
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", 5433)),
        user=os.getenv("PG_USER", "novus"),
        password=os.getenv("PG_PASSWORD", "novus123"),
        dbname=os.getenv("PG_DATABASE", "novus_kb"),
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT doc_id, chunk_index, content FROM chunks ORDER BY doc_id, chunk_index")
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {"id": f"{row[0]}:{row[1]}", "doc_id": row[0], "chunk_index": row[1], "content": row[2]}
        for row in rows
    ]


if __name__ == "__main__":
    print("Loading chunks from pgvector…")
    try:
        chunks = load_chunks_from_db()
        print(f"  Loaded {len(chunks)} chunks\n")
    except Exception as e:
        print(f"  DB unavailable ({e}). Using synthetic chunks for demo.\n")
        chunks = [
            {"id": "01_return_policy:0", "doc_id": "01_return_policy", "chunk_index": 0,
             "content": "Our return policy allows customers to get their money back within 30 days."},
            {"id": "02_membership_tiers:0", "doc_id": "02_membership_tiers", "chunk_index": 0,
             "content": "Premium Gold members receive 4 airport lounge visits per year."},
            {"id": "03_shipping:0", "doc_id": "03_shipping", "chunk_index": 0,
             "content": "Orders are shipped within 2 business days. Tracking numbers are emailed."},
        ]

    # A2.1 deliverable: vocabulary mismatch query
    query = "how do I get my money back"
    print(f"Query: '{query}'")
    print("-" * 60)
    results = bm25_simple(chunks, query)

    if not results:
        print("  No matching chunks found.")
    else:
        print(f"Top-{min(3, len(results))} BM25 results:\n")
        for i, chunk in enumerate(results[:3], 1):
            print(f"  [{i}] {chunk['doc_id']} (chunk {chunk.get('chunk_index', '?')})")
            print(f"       {chunk['content'][:120].strip()}…\n")

    # Check if the return/refund doc appeared
    top3_ids = [c["doc_id"] for c in results[:3]]
    return_docs = [d for d in top3_ids if "return" in d or "refund" in d or "account" in d]
    if return_docs:
        print(f"✅  Return/refund-related doc(s) in top-3: {return_docs}")
    else:
        print("⚠️  No return/refund doc in top-3 — vocabulary mismatch present")
        print(f"    Top-3 docs: {top3_ids}")
