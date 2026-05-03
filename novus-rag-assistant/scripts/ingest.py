"""
ingest.py — Chunk and embed all 19 Novus Bank policy documents, then store
            in pgvector.

Week 2 adds --strategy so all three chunking strategies can be compared via
the eval harness without changing code:

    python scripts/ingest.py --strategy fixed_size      # Week 1 baseline
    python scripts/ingest.py --strategy sliding_window  # A1.1
    python scripts/ingest.py --strategy sentence_aware  # A1.2

Run: python scripts/ingest.py
Re-run: safe — clears existing rows for each doc before inserting fresh ones.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Iterator

import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 500   # characters per chunk
BATCH_SIZE = 20    # embeddings per API call
CORPUS_DIR = Path(__file__).parent.parent / "corpus"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Chunking — delegates to chunker.py
# ---------------------------------------------------------------------------

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.chunker import chunk_text  # noqa: E402


def iter_documents(corpus_dir: Path) -> Iterator[tuple[str, str]]:
    """Yield (doc_id, content) for every .md file in corpus_dir, sorted."""
    for path in sorted(corpus_dir.glob("*.md")):
        doc_id = path.stem  # e.g. "03_debit_card_policy"
        content = path.read_text(encoding="utf-8")
        yield doc_id, content


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns list of 1536-dim float vectors."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", 5433)),
        user=os.getenv("PG_USER", "novus"),
        password=os.getenv("PG_PASSWORD", "novus123"),
        dbname=os.getenv("PG_DATABASE", "novus_kb"),
    )


def clear_doc(cur, doc_id: str):
    cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))


def insert_chunks(cur, doc_id: str, chunks: list[str], embeddings: list[list[float]]):
    for idx, (text, vec) in enumerate(zip(chunks, embeddings)):
        cur.execute(
            "INSERT INTO chunks (doc_id, chunk_index, content, embedding) "
            "VALUES (%s, %s, %s, %s)",
            (doc_id, idx, text, vec),
        )


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------

def ingest(strategy: str = "fixed_size"):
    conn = get_conn()
    conn.autocommit = False

    total_docs = 0
    total_chunks = 0

    for doc_id, content in iter_documents(CORPUS_DIR):
        chunks = chunk_text(content, strategy=strategy)
        if not chunks:
            continue

        # Embed in batches to respect API limits
        all_embeddings: list[list[float]] = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            embeddings = embed_batch(batch)
            all_embeddings.extend(embeddings)
            # Brief pause to avoid rate-limit spikes on first run
            if i + BATCH_SIZE < len(chunks):
                time.sleep(0.1)

        with conn.cursor() as cur:
            clear_doc(cur, doc_id)  # idempotent re-run support
            insert_chunks(cur, doc_id, chunks, all_embeddings)

        conn.commit()
        total_docs += 1
        total_chunks += len(chunks)
        print(f"  ✓ {doc_id}: {len(chunks)} chunks")

    conn.close()
    print(f"\n✅  Ingestion complete: {total_docs} docs, {total_chunks} chunks stored")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Novus Bank corpus into pgvector")
    parser.add_argument(
        "--strategy",
        choices=["fixed_size", "sliding_window", "sentence_aware"],
        default="fixed_size",
        help="Chunking strategy (default: fixed_size)",
    )
    args = parser.parse_args()
    print(f"\nIngesting with strategy: {args.strategy}\n")
    ingest(strategy=args.strategy)
