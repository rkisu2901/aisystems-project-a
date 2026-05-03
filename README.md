# Project A — Multi-Source Knowledge Base Assistant

Part of **AI Systems in Production** — AI Classroom Cohort 3

A production-grade RAG system built layer by layer across 4 weeks. Answers questions over a messy, realistic document corpus — internal docs, PDFs with tables, Markdown files, and support tickets.

## Setup

### 1. Prerequisites
- Python 3.11+
- Docker Desktop
- OpenAI API key (with credits)
- LangFuse account (cloud.langfuse.com — free tier)

### 2. Environment

```bash
cp .env.example .env
# Fill in your API keys in .env

docker-compose up -d

pip install -r requirements.txt
```

### 3. Run

```bash
# Set up the database
python scripts/setup_db.py

# Ingest documents
python scripts/ingest.py

# Test the pipeline
python scripts/rag.py

# Interactive demo
python scripts/demo.py
```

## Repo Structure

```
project-a/
├── corpus/              # Acmera company documents (19 files)
├── scripts/
│   ├── setup_db.py      # Create pgvector table + HNSW index
│   ├── ingest.py        # Chunk + embed + store documents
│   ├── rag.py           # Core RAG pipeline with LangFuse tracing
│   ├── demo.py          # Interactive query CLI
│   └── eval_harness.py  # Evaluation skeleton (built in Session 1)
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## What We Build (Week by Week)

| Week | Layer | What Gets Added |
|------|-------|-----------------|
| 1 | Evaluate | Golden dataset, eval harness, LLM-as-judge, LangFuse scores |
| 2 | Retrieve | Hybrid search, re-ranking, context engineering |
| 3 | Optimize & Observe | Semantic caching, model routing, full tracing |
| 4 | Harden & Deploy | Guardrails, structured outputs, AWS ECS Fargate |
