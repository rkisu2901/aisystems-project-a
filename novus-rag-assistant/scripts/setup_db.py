"""
setup_db.py — Create the pgvector schema for Novus Bank knowledge base.

Run once after `docker-compose up -d`. Safe to re-run (idempotent).
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    doc_id      TEXT    NOT NULL,          -- e.g. "03_debit_card_policy"
    chunk_index INTEGER NOT NULL,          -- 0-based position within the doc
    content     TEXT    NOT NULL,
    embedding   vector(1536)               -- text-embedding-3-small dimensionality
);

-- HNSW index: fast approximate nearest-neighbour search.
-- m=16 and ef_construction=64 are solid starting-point values for <50k chunks.
-- cosine distance matches how we query (1 - cosine similarity).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
"""


def setup():
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", 5433)),
        user=os.getenv("PG_USER", "novus"),
        password=os.getenv("PG_PASSWORD", "novus123"),
        dbname=os.getenv("PG_DATABASE", "novus_kb"),
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.close()
    print("✅  Schema ready: extension=vector, table=chunks, index=hnsw")


if __name__ == "__main__":
    setup()
