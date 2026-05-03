# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

Novus Bank RAG Assistant — Week 1 of AIC AI Systems Production Engineering. The explicit goal of this week is **measurement, not optimisation**: build an honest baseline, score it on 4 metrics, save it, and identify failure modes. Week 2 improvements are validated against `scripts/baseline_scores.json`.

## Commands

All scripts must be run from the **project root** (not from `scripts/`). `eval_harness.py` uses `sys.path.insert(0, parent)` to import `scripts.rag`, so running from a subdirectory breaks the import.

```bash
# Infrastructure — start both before running any script
docker-compose up -d                    # pgvector on port 5433
cd langfuse && docker-compose up -d     # Langfuse on port 3000 (optional, can skip)

# One-time setup (run in order)
python scripts/setup_db.py              # creates chunks table + HNSW index
python scripts/ingest.py                # chunks, embeds, stores all 19 corpus docs (~2 min)

# Interactive Q&A
python scripts/demo.py                                          # interactive mode
python scripts/demo.py --query "What is the UPI daily limit?"  # single query
python scripts/demo.py --demo                                   # 5 built-in demo queries

# Evaluation
python scripts/eval_harness.py                     # run all 55 queries, print scorecard
python scripts/eval_harness.py --limit 5           # quick smoke-test (5 queries, ~1 min)
python scripts/eval_harness.py --save-baseline     # run + save baseline_scores.json

# Regression check (compare current eval_results.json against saved baseline)
python scripts/check_regression.py
python scripts/check_regression.py --threshold 0.03   # tighter 3pp gate

# Synthetic data generation
python scripts/synthetic_generator.py --doc 02_membership_tiers.md --persona frustrated
python scripts/synthetic_generator.py --all --persona all --critique --output extra.json
```

## Architecture

### Data flow (three phases)

```
PHASE 1 — INGEST (run once)
  corpus/*.md  →  ingest.py  →  chunks table in pgvector (port 5433)

PHASE 2 — ANSWER (per query)
  question  →  rag.ask()  →  answer + retrieved_chunks + trace_id

PHASE 3 — EVALUATE (weekly, or after any pipeline change)
  golden_dataset.json  →  eval_harness.py  →  eval_results.json + baseline_scores.json
```

### RAG pipeline (`scripts/rag.py`)

The single public entry point is `ask(query, top_k=5) → dict`. Internally it chains four `@observe`-decorated functions:

1. `embed_query` — OpenAI `text-embedding-3-small`, returns 1536-dim vector
2. `retrieve` — pgvector cosine search via `<=>` operator, returns top-k dicts `{doc_id, chunk_index, content, similarity}`
3. `assemble_context` — formats chunks as numbered passages with source and score
4. `generate` — `gpt-4o-mini` at `temperature=0.1`, strict system prompt (`ONLY the context`)

Key constants at top of file: `TOP_K=5`, `EMBED_MODEL`, `CHAT_MODEL`, `TEMPERATURE`.

LangFuse is optional: if `LANGFUSE_*` keys are absent or the package isn't installed, a no-op `@observe` decorator is swapped in silently and `trace_id` is `None`.

### Evaluation metrics and normalisation (`scripts/eval_harness.py`)

Two retrieval metrics (already 0–1) and two LLM-judge metrics (1–5 scale):

| Metric | Range | How scored |
|---|---|---|
| `hit_rate` | 0–1 | 1 if `expected_source` in any retrieved `doc_id` |
| `mrr` | 0–1 | `1/rank` of first correct doc, else 0 |
| `faithfulness` | 1–5 | GPT-4o-mini judge at `temperature=0.0` |
| `correctness` | 1–5 | GPT-4o-mini judge vs `expected_answer` at `temperature=0.0` |

**Critical**: `check_regression.py` normalises `faithfulness` and `correctness` to 0–1 (`score / 5`) so that the 5pp threshold applies uniformly across all four metrics. The raw 1–5 values are stored in `eval_results.json` under `faithfulness_raw` / `correctness_raw`.

### Chunking strategy (intentional Week 1 weakness)

`ingest.py` uses fixed-size chunking: `CHUNK_SIZE=500` characters, **no overlap**, `BATCH_SIZE=20` per embedding API call. This is a deliberate baseline — `scripts/failure_taxonomy.md` documents the 5 resulting failure modes and Week 2 fixes. Do not add overlap until Week 2 or it breaks the baseline comparison.

`ingest.py` is **idempotent**: it calls `clear_doc()` before inserting, so re-running it is always safe.

### Two separate Docker stacks

| Compose file | Service | External port | Purpose |
|---|---|---|---|
| `docker-compose.yml` | `novus-rag-pgvector` | **5433** | pgvector (RAG knowledge base) |
| `langfuse/docker-compose.yml` | `langfuse-web` | **3000** | Langfuse dashboard + SDK API |
| `langfuse/docker-compose.yml` | `postgres` | 127.0.0.1:**5432** | Langfuse's own DB (not the RAG DB) |
| `langfuse/docker-compose.yml` | `minio` | **9090** | blob storage |

Port 5432 (Langfuse Postgres) and 5433 (pgvector) are intentionally different — no conflict.

### File lifecycle (what is committed vs gitignored)

| File | Committed? | Notes |
|---|---|---|
| `scripts/golden_dataset.json` | ✅ yes | Ground truth — 55 hand-written Q&A pairs; never auto-generate over this |
| `scripts/baseline_scores.json` | ✅ yes | Saved Week 1 benchmark; `check_regression.py` compares against this |
| `scripts/failure_taxonomy.md` | ✅ yes | Documents 5 failure modes + Week 2 fix strategy |
| `scripts/eval_results.json` | ❌ gitignored | Output of the latest `eval_harness.py` run |
| `.env` | ❌ gitignored | OpenAI + pgvector + Langfuse keys |
| `langfuse/.env` | ❌ gitignored | Langfuse internal service secrets |

### Corpus

19 Markdown files in `corpus/`, numbered `01_` – `19_`. File `19_internal_agent_guidelines.md` is marked CONFIDENTIAL in its content (internal agent rules). The `doc_id` stored in the DB is the filename stem (e.g. `12_fraud_and_dispute_policy`). `expected_source` in `golden_dataset.json` must match these stems exactly for `check_retrieval_hit` to work.

### Langfuse self-hosted setup

Dashboard: `http://localhost:3000` — login `admin@novusbank.local` / `NovusAdmin2026!`.  
The project API keys in `langfuse/.env` are pre-seeded and already written to the root `.env` — no manual key-copying needed after first boot. Wait ~2 min after `docker-compose up -d` for the `langfuse-web` container to log `Ready`.
