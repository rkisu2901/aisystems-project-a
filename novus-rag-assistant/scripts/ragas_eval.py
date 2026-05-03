"""
ragas_eval.py — RAGAS-based evaluation for the Novus Bank RAG pipeline.

A3.1 deliverable (CORE):
  Run the 10 hardest golden queries through RAGAS. Collect faithfulness,
  answer_relevancy, and context_precision. Produce a side-by-side table
  comparing LLM-as-judge scores vs RAGAS scores.

A3.3 deliverable (CHALLENGE):
  Add context_recall (requires ground_truth field). Compare against the
  manual hit_rate metric — they measure related but distinct things.

Install deps:
    pip install ragas datasets

Usage:
    python scripts/ragas_eval.py                        # all 10 hard queries
    python scripts/ragas_eval.py --limit 5              # quick test
    python scripts/ragas_eval.py --save results.json    # persist output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.rag import ask
from scripts.eval_harness import (
    check_retrieval_hit,
    calculate_mrr,
    judge_faithfulness,
    judge_correctness,
)

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"

# ---------------------------------------------------------------------------
# Select the 10 hardest queries from the golden dataset
# ---------------------------------------------------------------------------

def load_hard_queries(dataset_path: Path, limit: int | None = None) -> list[dict]:
    """Return up to `limit` hard/medium queries sorted by difficulty then id."""
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    # Prefer hard entries, then medium — these stress-test the pipeline most
    hard = [e for e in dataset if e.get("difficulty") == "hard"]
    medium = [e for e in dataset if e.get("difficulty") == "medium"]
    selected = (hard + medium)[:10]
    if limit:
        selected = selected[:limit]
    return selected


# ---------------------------------------------------------------------------
# RAGAS evaluation (A3.1)
# ---------------------------------------------------------------------------

def run_ragas_eval(entries: list[dict], use_hybrid: bool = False) -> dict:
    """Run RAGAS metrics on the selected entries.

    Returns a dict with:
        ragas_scores  — per-entry RAGAS scores
        our_scores    — per-entry LLM-as-judge scores (for side-by-side)
        comparison    — merged table rows
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
    except ImportError:
        print("❌  ragas / datasets not installed. Run: pip install ragas datasets")
        sys.exit(1)

    print(f"Running RAG pipeline on {len(entries)} queries…\n")
    rag_results = []
    our_scores = []

    for i, entry in enumerate(entries):
        print(f"  [{i+1}/{len(entries)}] {entry['id']}: {entry['query'][:55]}…")
        result = ask(entry["query"], use_hybrid=use_hybrid)
        faith = judge_faithfulness(result["answer"], result["context"])
        correct = judge_correctness(entry["query"], result["answer"], entry["expected_answer"])
        hit = check_retrieval_hit(result["retrieved_chunks"], entry["expected_source"])
        mrr = calculate_mrr(result["retrieved_chunks"], entry["expected_source"])

        rag_results.append({
            "question": entry["query"],
            "answer": result["answer"],
            "contexts": [result["context"]],
            "ground_truth": entry["expected_answer"],
        })
        our_scores.append({
            "id": entry["id"],
            "query": entry["query"],
            "expected_source": entry["expected_source"],
            "hit": hit,
            "mrr": mrr,
            "our_faithfulness": faith["score"],
            "our_faithfulness_reason": faith["reason"],
            "our_correctness": correct["score"],
            "our_correctness_reason": correct["reason"],
            "answer": result["answer"],
        })

    print("\nRunning RAGAS evaluation…\n")
    dataset = Dataset.from_list(rag_results)

    # A3.1: faithfulness, answer_relevancy, context_precision
    # A3.3: context_recall (requires ground_truth — already added above)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    ragas_df = scores.to_pandas()

    # Build side-by-side comparison table
    comparison = []
    for i, entry in enumerate(entries):
        row = {
            "id": entry["id"],
            "query": entry["query"][:50],
            # Our scores (normalized to 0-1 for fair comparison)
            "our_faithfulness_1to5": our_scores[i]["our_faithfulness"],
            "our_faithfulness_0to1": round(our_scores[i]["our_faithfulness"] / 5, 3),
            "our_correctness_1to5": our_scores[i]["our_correctness"],
            "our_correctness_0to1": round(our_scores[i]["our_correctness"] / 5, 3),
            "our_hit_rate": int(our_scores[i]["hit"]),
            "our_mrr": round(our_scores[i]["mrr"], 3),
            # RAGAS scores (all 0-1)
            "ragas_faithfulness": round(float(ragas_df["faithfulness"][i]), 3),
            "ragas_answer_relevancy": round(float(ragas_df["answer_relevancy"][i]), 3),
            "ragas_context_precision": round(float(ragas_df["context_precision"][i]), 3),
            "ragas_context_recall": round(float(ragas_df["context_recall"][i]), 3),
        }
        comparison.append(row)

    return {
        "ragas_scores": ragas_df.to_dict(orient="records"),
        "our_scores": our_scores,
        "comparison": comparison,
        "ragas_summary": {
            "faithfulness": round(ragas_df["faithfulness"].mean(), 3),
            "answer_relevancy": round(ragas_df["answer_relevancy"].mean(), 3),
            "context_precision": round(ragas_df["context_precision"].mean(), 3),
            "context_recall": round(ragas_df["context_recall"].mean(), 3),
        },
        "our_summary": {
            "faithfulness_norm": round(
                sum(s["our_faithfulness"] for s in our_scores) / (len(our_scores) * 5), 3
            ),
            "correctness_norm": round(
                sum(s["our_correctness"] for s in our_scores) / (len(our_scores) * 5), 3
            ),
            "hit_rate": round(sum(s["hit"] for s in our_scores) / len(our_scores), 3),
            "mrr": round(sum(s["mrr"] for s in our_scores) / len(our_scores), 3),
        },
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_comparison(result: dict) -> None:
    comparison = result["comparison"]
    our_sum = result["our_summary"]
    ragas_sum = result["ragas_summary"]

    print("\n" + "=" * 110)
    print("  RAGAS vs LLM-as-Judge Side-by-Side Comparison")
    print("=" * 110)
    header = (
        f"{'ID':<10} {'Our Faith':>10} {'RAGAS Faith':>12} "
        f"{'Our Corr':>9} {'RAGAS Rel':>10} "
        f"{'Our Hit':>8} {'RAGAS CP':>9} {'RAGAS CR':>9}"
    )
    print(header)
    print("-" * 110)
    for row in comparison:
        print(
            f"{row['id']:<10} "
            f"{row['our_faithfulness_0to1']:>10.3f} "
            f"{row['ragas_faithfulness']:>12.3f} "
            f"{row['our_correctness_0to1']:>9.3f} "
            f"{row['ragas_answer_relevancy']:>10.3f} "
            f"{row['our_hit_rate']:>8} "
            f"{row['ragas_context_precision']:>9.3f} "
            f"{row['ragas_context_recall']:>9.3f}"
        )
    print("-" * 110)
    print(
        f"{'MEAN':<10} "
        f"{our_sum['faithfulness_norm']:>10.3f} "
        f"{ragas_sum['faithfulness']:>12.3f} "
        f"{our_sum['correctness_norm']:>9.3f} "
        f"{ragas_sum['answer_relevancy']:>10.3f} "
        f"{our_sum['hit_rate']:>8.3f} "
        f"{ragas_sum['context_precision']:>9.3f} "
        f"{ragas_sum['context_recall']:>9.3f}"
    )
    print("=" * 110)

    print("""
Columns:
  Our Faith     — LLM-as-judge faithfulness (1–5 normalised to 0–1)
  RAGAS Faith   — RAGAS faithfulness (0–1) using NLI entailment
  Our Corr      — LLM-as-judge correctness vs expected_answer (0–1)
  RAGAS Rel     — RAGAS answer_relevancy: semantic similarity to question
  Our Hit       — 1 if expected source doc in top-5 retrieved chunks
  RAGAS CP      — context_precision: relevant chunks ranked higher
  RAGAS CR      — context_recall: retrieved context covers ground_truth

A3.1 Reflection (150 words):
  RAGAS measures faithfulness through NLI entailment checking rather than
  asking an LLM to score a rubric. This is more objective — it directly
  checks whether each claim in the answer is supported by the context using
  a trained inference model, rather than relying on the same LLM that
  generated the answer. RAGAS answer_relevancy differs fundamentally from
  our correctness judge: relevancy measures how well the answer addresses
  the question (is it on-topic?), while our correctness compares content
  against a gold-standard expected answer. A verbose but off-topic answer
  scores low on relevancy but could score high on our judge if it happens
  to include the right facts. context_precision rewards systems that rank
  relevant chunks higher; context_recall (A3.3) checks whether the retrieved
  context contains enough information to construct the ground_truth answer.
  Hit rate asks a binary question (is the right doc present?); context_recall
  asks a content question (is the right information extractable from what
  was retrieved?). A chunk can be from the correct document but still fail
  context_recall if the relevant sentence was split at a chunk boundary.
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAGAS eval for Novus Bank RAG")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N hard queries")
    parser.add_argument("--save", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid retrieval")
    args = parser.parse_args()

    entries = load_hard_queries(DATASET_PATH, limit=args.limit)
    print(f"Selected {len(entries)} hard/medium queries for RAGAS evaluation\n")

    t0 = time.time()
    result = run_ragas_eval(entries, use_hybrid=args.hybrid)
    elapsed = round(time.time() - t0, 1)

    print_comparison(result)
    print(f"Total evaluation time: {elapsed}s")

    if args.save:
        save_path = Path(args.save)
        save_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Results saved → {save_path}")


if __name__ == "__main__":
    main()
