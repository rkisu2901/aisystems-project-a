# Novus Bank Knowledge Base — RAG Assistant

> **Week 1 of AI Systems Production Engineering**
> Building the measurement foundation before we can improve anything.

---

## The Use Case

Novus Bank is a fictional Indian neo-bank with 19 policy documents covering everything from account opening to fraud liability. Customer support agents field hundreds of questions daily about these policies — interest rates, fee structures, liability rules — and the policies themselves are dense, interlinked, and full of edge cases.

This project builds a **Retrieval-Augmented Generation (RAG)** pipeline that lets a support agent (or an automated system) ask plain-English questions and get grounded, policy-accurate answers. Every answer cites which document it came from. Every run is traced in LangFuse.

The Week 1 goal is not to build the best RAG pipeline. It is to build an **honest, reproducible measurement system** so we know exactly what is broken before we try to fix anything in Week 2.

---

## Repository Structure

```
novus-rag-assistant/
├── corpus/                          # 19 Novus Bank policy documents (.md)
│   ├── 01_account_opening_policy.md
│   ├── 02_membership_tiers.md
│   ├── ...
│   └── 19_internal_agent_guidelines.md
├── scripts/
│   ├── setup_db.py                  # Create pgvector schema (run once)
│   ├── ingest.py                    # Chunk + embed + store all 19 docs
│   ├── rag.py                       # Core RAG pipeline (embed → retrieve → generate)
│   ├── demo.py                      # Rich CLI for interactive Q&A
│   ├── golden_dataset.json          # 55 hand-written evaluation queries
│   ├── eval_harness.py              # 4-metric eval: hit rate, MRR, faithfulness, correctness
│   ├── synthetic_generator.py       # Generate synthetic Q&A pairs (3 personas + critique)
│   ├── check_regression.py          # CI-style regression checker (5pp threshold)
│   ├── failure_taxonomy.md          # Analysis of 5 failure modes in the baseline
│   ├── eval_results.json            # Latest eval run output (git-ignored)
│   └── baseline_scores.json        # Saved baseline for regression checks
├── docker-compose.yml               # pgvector on port 5433
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Prerequisites

```bash
# Start the database
docker-compose up -d

# Install Python dependencies
pip install -r requirements.txt

# Copy and fill in your API keys
cp .env.example .env
# Edit .env: add OPENAI_API_KEY and optionally LANGFUSE_* keys
```

### 2. Set Up the Database

```bash
python scripts/setup_db.py
# ✅  Schema ready: extension=vector, table=chunks, index=hnsw
```

### 3. Ingest All 19 Documents

```bash
python scripts/ingest.py
# ✓ 01_account_opening_policy: 14 chunks
# ✓ 02_membership_tiers: 11 chunks
# ...
# ✅  Ingestion complete: 19 docs, ~220 chunks stored
```

### 4. Try a Question

```bash
python scripts/demo.py --query "What is my liability if I report fraud after 5 days?"
```

### 5. Run the Full Evaluation

```bash
python scripts/eval_harness.py --save-baseline
```

Expected output:
```
==================================================
  Novus Bank RAG — Overall Scorecard
==================================================
  Queries evaluated : 55
  Hit Rate          : ~82%
  MRR               : ~0.72
  Faithfulness      : ~3.8/5.0  (~76%)
  Correctness       : ~3.5/5.0  (~70%)
==================================================
```

*(Actual numbers will vary by model version and embedding space. These are representative Week 1 baseline figures.)*

---

## The Pipeline

```
Question
   │
   ▼
embed_query()          ← text-embedding-3-small, 1536 dims
   │
   ▼
retrieve()             ← cosine similarity search via pgvector HNSW index, top-5
   │
   ▼
assemble_context()     ← numbered passages with doc_id and similarity score
   │
   ▼
generate()             ← gpt-4o-mini, temperature=0.1, grounded system prompt
   │
   ▼
Answer + Sources + Trace ID
```

Every function is decorated with `@observe` from LangFuse, giving you a full span tree per query with latency and token counts at each step.

---

## The Golden Dataset

`scripts/golden_dataset.json` contains **55 hand-written queries** covering all 12 policy categories:

| Category | Count | Sample Query |
|----------|-------|-------------|
| payments | 14 | "What is the daily UPI limit?" |
| account | 12 | "How do I reactivate a dormant account?" |
| products | 7 | "What is the FD rate for Elite senior citizens?" |
| loans | 5 | "Can I prepay my personal loan before 6 EMIs?" |
| fraud | 5 | "I shared my OTP — am I still covered?" |
| membership | 4 | "What AQB is needed for Novus Plus?" |
| grievance | 3 | "When can I go to the RBI Ombudsman?" |
| ... | ... | ... |

Difficulty distribution: ~60% easy, ~25% medium, ~15% hard. Hard queries require reasoning across multiple conditions (tier + seniority + tenure) or cover edge cases with non-obvious liability rules.

---

## Evaluation Metrics

| Metric | What it measures | How computed |
|--------|-----------------|-------------|
| **Hit Rate** | Is the right doc in top-5? | `1` if expected_source in retrieved doc_ids |
| **MRR** | How high does the right doc rank? | `1/rank` if found, `0` if not in top-5 |
| **Faithfulness** | Does the answer stay grounded? | GPT-4o-mini judge, 1–5 scale, temp=0.0 |
| **Correctness** | Is the answer factually right? | GPT-4o-mini judge vs expected_answer, 1–5 |

Faithfulness and Correctness are normalized to 0–1 before aggregation so all four metrics are on the same scale for the regression checker.

---

## Known Weaknesses (Week 1 Baseline)

These are **intentional** — the point is to measure them, not hide them.

1. **Boundary splits** — 500-char fixed chunking with no overlap. Answers that span a chunk boundary will be incomplete.
2. **Table row separation** — Markdown tables get split mid-row. Tier-specific numbers (e.g., Elite FD rate) may not be in the same chunk as the table header.
3. **Multi-hop queries** — Queries that need two documents (e.g., "Can I use my loan against my tax-saving FD?") will fail because we do a single retrieval pass.
4. **Vocabulary overlap** — "grace period" appears in 3 different policy documents; the wrong one may rank higher.

See `scripts/failure_taxonomy.md` for detailed analysis of each failure mode and the Week 2 fix strategy.

---

## Synthetic Data Generator

```bash
# Generate 5 Q&A pairs from the membership tiers doc, frustrated persona
python scripts/synthetic_generator.py --doc 02_membership_tiers.md --persona frustrated

# Run with auto-critique (drops pairs scoring below 3/5)
python scripts/synthetic_generator.py --doc 12_fraud_and_dispute_policy.md --critique

# Generate across all docs, all 3 personas, save output
python scripts/synthetic_generator.py --all --persona all --critique --output extra_pairs.json
```

The three personas surface different failure modes:
- `standard` — clean queries, tests basic retrieval accuracy
- `frustrated` — emotional framing, tests whether the LLM stays on-policy
- `mismatch` — wrong assumptions, tests whether the LLM corrects the user

---

## Regression Checker

After any pipeline change, run:

```bash
python scripts/check_regression.py
```

Fails with exit code 1 if any metric drops more than 5 percentage points from the saved baseline. Designed to run in CI on every pull request that touches `rag.py`, `ingest.py`, or the system prompt.

---

## LangFuse Observability

With `LANGFUSE_*` keys in your `.env`, every eval run posts:
- A trace per query with span-level latency for `embed_query`, `retrieve`, `assemble_context`, `generate`
- Faithfulness and Correctness scores attached to each trace
- Filterable view: "show me all traces where correctness < 3" → these are your failure cases

If LangFuse keys are not set, the pipeline runs normally — it just skips the telemetry.

---

## What's Next (Week 2 Preview)

The measurement baseline tells us exactly where to look:

1. Switch to 200-char overlapping chunks → fixes boundary split failures
2. Table-aware chunking → fixes tier conflation failures
3. Metadata pre-filtering by policy domain → fixes vocabulary overlap failures
4. HyDE (Hypothetical Document Embeddings) → fixes wrong-ranking failures

Every change will be validated against the `baseline_scores.json` saved this week. If hit rate doesn't improve, the change ships without that hypothesis.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required. OpenAI API key. |
| `PG_HOST` | Database host (default: localhost) |
| `PG_PORT` | Database port (default: 5433) |
| `PG_USER` | Database user (default: novus) |
| `PG_PASSWORD` | Database password (default: novus123) |
| `PG_DATABASE` | Database name (default: novus_kb) |
| `LANGFUSE_PUBLIC_KEY` | Optional. LangFuse project public key. |
| `LANGFUSE_SECRET_KEY` | Optional. LangFuse project secret key. |
| `LANGFUSE_HOST` | Optional. LangFuse host (default: cloud.langfuse.com) |
