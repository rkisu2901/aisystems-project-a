"""
eval_harness.py — Evaluation harness for the Novus Bank RAG pipeline.

Metrics computed:
  1. Hit Rate     — Is the expected source document in the top-5 retrieved chunks?
  2. MRR          — Mean Reciprocal Rank of the expected source within top-5.
  3. Faithfulness — LLM-as-judge: does the answer stay grounded in the context? (1-5)
  4. Correctness  — LLM-as-judge: does the answer correctly address the question? (1-5)

The LLM-as-judge pattern uses GPT-4o-mini at temperature=0 for deterministic scoring.
Each judge returns {"score": int, "reason": str} — the reason is stored for failure
analysis but not aggregated numerically.

Usage:
    python scripts/eval_harness.py                            # local, all 57 queries
    python scripts/eval_harness.py --save-baseline            # also saves baseline_scores.json
    python scripts/eval_harness.py --limit 10                 # quick smoke-test on first 10
    python scripts/eval_harness.py --use-hybrid               # hybrid BM25+dense+RRF retrieval
    python scripts/eval_harness.py --remote --api-url https://<alb-dns>   # remote deployed API (D4.1)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.rag import ask

try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from langfuse import Langfuse
    LANGFUSE_ENABLED = True
    _lf = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
except Exception:
    LANGFUSE_ENABLED = False
    _lf = None

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"
BASELINE_PATH = Path(__file__).parent / "baseline_scores.json"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
JUDGE_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Metric 1: Retrieval Hit Rate
# ---------------------------------------------------------------------------

def check_retrieval_hit(retrieved_chunks: list[dict], expected_source: str) -> bool:
    """Return True if expected_source appears in any of the retrieved chunks.

    We check if the expected doc_id is a substring match — e.g.
    expected_source="03_debit_card_policy" will match doc_id="03_debit_card_policy".
    """
    return any(expected_source in chunk["doc_id"] for chunk in retrieved_chunks)


# ---------------------------------------------------------------------------
# Metric 2: Mean Reciprocal Rank
# ---------------------------------------------------------------------------

def calculate_mrr(retrieved_chunks: list[dict], expected_source: str) -> float:
    """Return 1/rank if the expected source is in retrieved_chunks, else 0.

    Rank is 1-based. If the correct doc appears at position 2, MRR = 0.5.
    If it doesn't appear at all in top-k, MRR = 0.
    """
    for rank, chunk in enumerate(retrieved_chunks, 1):
        if expected_source in chunk["doc_id"]:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Metric 3: Faithfulness (LLM-as-judge)
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """You are an evaluation judge for a banking AI assistant.

Score the following answer on FAITHFULNESS: does the answer stay grounded in
the provided context, or does it add information not present in the context?

Scoring rubric:
  5 — Every claim in the answer is directly supported by the context.
  4 — Mostly grounded; minor paraphrasing that doesn't change meaning.
  3 — Partly grounded; some claims go slightly beyond the context.
  2 — Significant information added beyond what the context states.
  1 — Answer largely fabricated or contradicts the context.

Context:
{context}

Answer:
{answer}

Respond with valid JSON only, no other text:
{{"score": <1-5>, "reason": "<one sentence explanation>"}}"""


def judge_faithfulness(answer: str, context: str) -> dict[str, Any]:
    """Score answer faithfulness relative to the retrieved context (1-5)."""
    prompt = FAITHFULNESS_PROMPT.format(context=context[:3000], answer=answer)
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Metric 4: Correctness (LLM-as-judge)
# ---------------------------------------------------------------------------

CORRECTNESS_PROMPT = """You are an evaluation judge for a banking AI assistant.

Score the following answer on CORRECTNESS: does it accurately and completely
address the customer's question, compared to the expected answer?

Scoring rubric:
  5 — Fully correct; all key facts match the expected answer.
  4 — Mostly correct; minor omission or slightly different phrasing.
  3 — Partially correct; gets the main point but misses important details.
  2 — Mostly wrong; addresses the question but with significant factual errors.
  1 — Completely wrong or doesn't address the question.

Question: {query}

Expected Answer: {expected_answer}

Actual Answer: {answer}

Respond with valid JSON only, no other text:
{{"score": <1-5>, "reason": "<one sentence explanation>"}}"""


def judge_correctness(query: str, answer: str, expected_answer: str) -> dict[str, Any]:
    """Score answer correctness against the expected answer (1-5)."""
    prompt = CORRECTNESS_PROMPT.format(
        query=query, expected_answer=expected_answer, answer=answer
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Optional: Attach LangFuse scores to a trace
# ---------------------------------------------------------------------------

def attach_langfuse_scores(trace_id: str, faithfulness: float, correctness: float) -> None:
    """Post faithfulness and correctness scores to a LangFuse trace.

    This links eval metrics to the exact trace in the LangFuse UI, enabling
    filtered views like "show me all traces where correctness < 3".
    """
    if not LANGFUSE_ENABLED or not _lf or not trace_id:
        return
    try:
        _lf.score(
            trace_id=trace_id,
            name="faithfulness",
            value=faithfulness,
        )
        _lf.score(
            trace_id=trace_id,
            name="correctness",
            value=correctness,
        )
    except Exception:
        pass  # LangFuse is observability-only; never block the eval loop


# ---------------------------------------------------------------------------
# D4.1 — Remote ask: call deployed API instead of local import
# ---------------------------------------------------------------------------

def ask_remote(query: str, api_url: str, mode: str = "dense") -> dict:
    """Call the deployed FastAPI /query endpoint and return a result dict
    shaped identically to the local ask() return value.

    Requires `requests` package and a running API at api_url.
    """
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("'requests' package required for --remote mode. pip install requests")

    response = _requests.post(
        f"{api_url.rstrip('/')}/query",
        json={"query": query, "mode": mode},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    return {
        "query": query,
        "answer": data["answer"],
        "retrieved_chunks": data.get("retrieved_chunks", []),
        "context": data.get("context", ""),
        "retrieval_mode": data.get("retrieval_mode", "remote"),
        "model_used": data.get("model_used", "unknown"),
        "cache_similarity": data.get("cache_similarity"),
        "trace_id": data.get("trace_id"),
        "elapsed_seconds": data.get("elapsed_seconds", 0.0),
    }


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(dataset: list[dict], verbose: bool = True, use_hybrid: bool = False,
             remote: bool = False, api_url: str | None = None) -> list[dict]:
    """Run the full eval pipeline over all dataset entries.

    Args:
        remote:   If True, call ask_remote() against the deployed API.
        api_url:  Required when remote=True. Base URL of the deployed API.

    Returns a list of per-entry result dicts with all metrics.
    """
    if remote and not api_url:
        raise ValueError("--api-url is required when --remote is set")

    results = []

    for i, entry in enumerate(dataset):
        if verbose:
            print(f"  [{i+1}/{len(dataset)}] {entry['id']}: {entry['query'][:60]}…")

        if remote:
            mode = "hybrid" if use_hybrid else "dense"
            try:
                rag_result = ask_remote(entry["query"], api_url, mode=mode)
            except Exception as exc:
                print(f"    [WARN] Remote call failed for {entry['id']}: {exc} — skipping")
                results.append({
                    "id": entry["id"], "query": entry["query"],
                    "category": entry.get("category", "unknown"),
                    "difficulty": entry.get("difficulty", "unknown"),
                    "expected_source": entry["expected_source"],
                    "answer": "ERROR", "retrieved_sources": [],
                    "hit": False, "mrr": 0.0,
                    "faithfulness": 1, "faithfulness_reason": "remote error",
                    "correctness": 1, "correctness_reason": "remote error",
                    "elapsed_seconds": 0.0, "trace_id": None,
                })
                continue
        else:
            rag_result = ask(entry["query"], use_hybrid=use_hybrid)

        hit = check_retrieval_hit(rag_result["retrieved_chunks"], entry["expected_source"])
        mrr = calculate_mrr(rag_result["retrieved_chunks"], entry["expected_source"])
        faith_result = judge_faithfulness(rag_result["answer"], rag_result["context"])
        correct_result = judge_correctness(
            entry["query"], rag_result["answer"], entry["expected_answer"]
        )

        if rag_result.get("trace_id"):
            attach_langfuse_scores(
                rag_result["trace_id"],
                faith_result["score"],
                correct_result["score"],
            )

        result = {
            "id": entry["id"],
            "query": entry["query"],
            "category": entry.get("category", "unknown"),
            "difficulty": entry.get("difficulty", "unknown"),
            "expected_source": entry["expected_source"],
            "answer": rag_result["answer"],
            "retrieved_sources": [c["doc_id"] for c in rag_result["retrieved_chunks"]],
            "hit": hit,
            "mrr": mrr,
            "faithfulness": faith_result["score"],
            "faithfulness_reason": faith_result["reason"],
            "correctness": correct_result["score"],
            "correctness_reason": correct_result["reason"],
            "elapsed_seconds": rag_result["elapsed_seconds"],
            "trace_id": rag_result.get("trace_id"),
        }
        results.append(result)

    return results


def compute_scorecard(results: list[dict]) -> dict:
    """Aggregate per-entry results into a summary scorecard."""
    n = len(results)
    if n == 0:
        return {}

    hit_rate = sum(r["hit"] for r in results) / n
    mrr = sum(r["mrr"] for r in results) / n
    # Normalize 1-5 scores to 0-1 for consistent comparison
    faithfulness = sum(r["faithfulness"] for r in results) / (n * 5)
    correctness = sum(r["correctness"] for r in results) / (n * 5)

    return {
        "n": n,
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "faithfulness": round(faithfulness, 4),   # normalized 0-1
        "correctness": round(correctness, 4),     # normalized 0-1
        "faithfulness_raw": round(sum(r["faithfulness"] for r in results) / n, 2),
        "correctness_raw": round(sum(r["correctness"] for r in results) / n, 2),
    }


def run_stratified_eval(results: list[dict]) -> dict[str, dict]:
    """Break down scores by category and difficulty.

    Returns {category: scorecard, ...} and {difficulty: scorecard, ...}.
    """
    by_category: dict[str, list] = {}
    by_difficulty: dict[str, list] = {}

    for r in results:
        by_category.setdefault(r["category"], []).append(r)
        by_difficulty.setdefault(r["difficulty"], []).append(r)

    return {
        "by_category": {cat: compute_scorecard(rows) for cat, rows in by_category.items()},
        "by_difficulty": {diff: compute_scorecard(rows) for diff, rows in by_difficulty.items()},
    }


def print_scorecard(scorecard: dict, title: str = "Overall Scorecard") -> None:
    width = 50
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    print(f"  Queries evaluated : {scorecard.get('n', 0)}")
    print(f"  Hit Rate          : {scorecard.get('hit_rate', 0):.1%}")
    print(f"  MRR               : {scorecard.get('mrr', 0):.4f}")
    print(f"  Faithfulness      : {scorecard.get('faithfulness_raw', 0):.2f}/5.0  ({scorecard.get('faithfulness', 0):.1%})")
    print(f"  Correctness       : {scorecard.get('correctness_raw', 0):.2f}/5.0  ({scorecard.get('correctness', 0):.1%})")
    print("=" * width + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Novus Bank RAG eval harness")
    parser.add_argument("--save-baseline", action="store_true", help="Save scores as baseline")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N entries")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-entry progress")
    parser.add_argument("--use-hybrid", action="store_true", help="Use hybrid BM25+dense+RRF retrieval (A2.3)")
    parser.add_argument("--remote", action="store_true", help="Call deployed API instead of local import (D4.1)")
    parser.add_argument("--api-url", type=str, default=None, help="Base URL of deployed API, e.g. https://<alb-dns>")
    args = parser.parse_args()

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]

    if args.remote:
        mode_label = f"REMOTE ({args.api_url})"
    elif args.use_hybrid:
        mode_label = "HYBRID (BM25+dense+RRF)"
    else:
        mode_label = "DENSE (pgvector only)"

    print(f"\nRunning eval on {len(dataset)} queries… [{mode_label}]\n")
    t0 = time.time()
    results = run_eval(
        dataset,
        verbose=not args.quiet,
        use_hybrid=args.use_hybrid,
        remote=args.remote,
        api_url=args.api_url,
    )
    elapsed = round(time.time() - t0, 1)

    scorecard = compute_scorecard(results)
    stratified = run_stratified_eval(results)

    print_scorecard(scorecard, title="Novus Bank RAG — Overall Scorecard")

    # Per-category breakdown
    print("Per-category breakdown:")
    cat_scores = stratified["by_category"]
    for cat in sorted(cat_scores, key=lambda c: cat_scores[c].get("correctness", 0)):
        sc = cat_scores[cat]
        print(
            f"  {cat:<20} n={sc['n']:>2}  hit={sc['hit_rate']:.0%}  "
            f"faith={sc['faithfulness_raw']:.1f}  correct={sc['correctness_raw']:.1f}"
        )

    # Per-difficulty breakdown
    print("\nPer-difficulty breakdown:")
    for diff in ["easy", "medium", "hard"]:
        if diff in stratified["by_difficulty"]:
            sc = stratified["by_difficulty"][diff]
            print(
                f"  {diff:<8} n={sc['n']:>2}  hit={sc['hit_rate']:.0%}  "
                f"correct={sc['correctness_raw']:.1f}"
            )

    print(f"\nTotal eval time: {elapsed}s")

    # Persist results
    output = {
        "scorecard": scorecard,
        "stratified": stratified,
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved → {RESULTS_PATH}")

    if args.save_baseline:
        baseline = {"scorecard": scorecard, "stratified": stratified}
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Baseline saved → {BASELINE_PATH}")


if __name__ == "__main__":
    main()
