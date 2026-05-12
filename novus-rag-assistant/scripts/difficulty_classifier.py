"""
difficulty_classifier.py — Query difficulty router for Novus Bank RAG pipeline.

Two classifiers:
  P1B.1 — classify_difficulty_regex(query) → score 1–5  (keyword/regex rules)
  P1B.2 — classify_difficulty_llm(query)   → score 1–5  (GPT-4o-mini)

Routing rule (both):
  score >= 4  →  gpt-4o      (complex, multi-condition, tier-specific)
  score <  4  →  gpt-4o-mini (simple factual lookup)

Usage:
    from scripts.difficulty_classifier import route_model_llm
    model, score = route_model_llm("What is the return window for Premium Gold?")
    # → ("gpt-4o", 4)

CLI demo:
    python scripts/difficulty_classifier.py --demo
    python scripts/difficulty_classifier.py --compare   # regex vs LLM on 10 queries
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_COMPLEX = "gpt-4o"
MODEL_SIMPLE  = "gpt-4o-mini"
SCORE_THRESHOLD = 4          # score >= threshold → complex model

# ---------------------------------------------------------------------------
# P1B.1 — Regex / keyword difficulty classifier
# ---------------------------------------------------------------------------

# Each pattern adds +1 to score (base score = 1, max = 5 via clamp).
_COMPLEX_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"premium\s+(gold|silver)", re.I),          "tier-specific query"),
    (re.compile(r"diwali|promotional|flash\s+sale", re.I),  "promotional edge case"),
    (re.compile(r"(corporate|bulk)\s+(order|return)", re.I),"multi-policy query"),
    (re.compile(r"\d+\s+(days?|items?|units?)", re.I),      "quantitative edge case"),
    (re.compile(r"difference\s+between", re.I),             "comparison query"),
]


def classify_difficulty_regex(query: str) -> int:
    """Score query difficulty 1–5 using keyword/regex rules.

    Base score = 1. Each matched pattern adds +1. Clamped to [1, 5].
    """
    score = 1
    for pattern, _ in _COMPLEX_PATTERNS:
        if pattern.search(query):
            score += 1
    return min(score, 5)


def route_model_regex(query: str) -> tuple[str, int]:
    """Return (model_name, score) using regex classification.

    score >= 4 → gpt-4o, else → gpt-4o-mini.
    """
    score = classify_difficulty_regex(query)
    model = MODEL_COMPLEX if score >= SCORE_THRESHOLD else MODEL_SIMPLE
    return model, score


def explain_regex(query: str) -> list[str]:
    """Return list of matched pattern labels (for debugging/demo output)."""
    matched = []
    for pattern, label in _COMPLEX_PATTERNS:
        if pattern.search(query):
            matched.append(label)
    return matched


# ---------------------------------------------------------------------------
# P1B.2 — LLM difficulty classifier
# ---------------------------------------------------------------------------

_LLM_PROMPT = """\
Rate this customer support query on a difficulty scale of 1-5.

1-2 = single fact lookup (return window, shipping cost, basic policy)
3   = one condition to evaluate (open vs closed electronics, age restrictions)
4-5 = multiple conditions, tier-specific rules, cross-document reasoning, or comparisons

Query: {query}

Respond ONLY with JSON: {{"score": N, "reason": "one line"}}"""


def classify_difficulty_llm(query: str) -> dict[str, Any]:
    """Score query difficulty 1–5 using GPT-4o-mini as judge.

    Returns {"score": int, "reason": str}.
    Falls back to regex score if the API call fails.
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        response = client.chat.completions.create(
            model=MODEL_SIMPLE,       # cheap model for the routing decision itself
            temperature=0,
            messages=[
                {"role": "user", "content": _LLM_PROMPT.format(query=query)}
            ],
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(raw)
        return {"score": int(result["score"]), "reason": result.get("reason", "")}
    except Exception as exc:
        fallback_score = classify_difficulty_regex(query)
        return {"score": fallback_score, "reason": f"LLM failed ({exc}), used regex fallback"}


def route_model_llm(query: str) -> tuple[str, int]:
    """Return (model_name, score) using LLM classification.

    score >= 4 → gpt-4o, else → gpt-4o-mini.
    """
    result = classify_difficulty_llm(query)
    score = result["score"]
    model = MODEL_COMPLEX if score >= SCORE_THRESHOLD else MODEL_SIMPLE
    return model, score


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

_DEMO_QUERIES = [
    "What is the standard return window?",
    "How much does shipping cost?",
    "What is the return policy for opened electronics?",
    "Can I get a bulk order discount?",
    "What is the difference between Premium Gold and Premium Silver membership?",
    "Is there a Diwali promotional discount on SIPs?",
    "What is the corporate return policy for 50 items?",
    "How do I track my order?",
    "Can I cancel within 24 hours?",
    "What are the NPS-linked FD rates for Premium Gold members with 30-day lock-in?",
]


def _print_table(rows: list[dict], title: str):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")
    print(f"{'#':<3} {'Query':<46} {'Score':>5}  {'Model'}")
    print("-" * 70)
    for r in rows:
        q = r["query"][:45] + ("…" if len(r["query"]) > 45 else "")
        print(f"{r['i']:<3} {q:<46} {r['score']:>5}  {r['model']}")


def _run_demo():
    rows = []
    for i, q in enumerate(_DEMO_QUERIES, 1):
        model, score = route_model_regex(q)
        rows.append({"i": i, "query": q, "score": score, "model": model})
    _print_table(rows, "P1B.1 — Regex Router (10 queries)")

    print("\n[2 examples where regex gets it wrong]")
    print("  • 'What is the 30-day return window?' → score 2 (matches \\d+ days) but it's a simple lookup")
    print("  • 'Difference in price between two items?' → score 2 (matches 'difference between') but not policy-complex")


def _run_compare():
    print("\nRunning LLM classifier on 10 queries (10 API calls)…")
    rows_regex, rows_llm = [], []
    for i, q in enumerate(_DEMO_QUERIES, 1):
        r_model, r_score = route_model_regex(q)
        l_result         = classify_difficulty_llm(q)
        l_score          = l_result["score"]
        l_model          = MODEL_COMPLEX if l_score >= SCORE_THRESHOLD else MODEL_SIMPLE
        rows_regex.append({"i": i, "query": q, "score": r_score, "model": r_model})
        rows_llm.append(  {"i": i, "query": q, "score": l_score, "model": l_model,
                           "reason": l_result["reason"]})

    _print_table(rows_regex, "P1B.1 — Regex Router")
    _print_table(rows_llm,   "P1B.2 — LLM Router")

    print("\n--- LLM reasons ---")
    for r in rows_llm:
        print(f"  [{r['i']}] {r['reason']}")

    # Cost analysis
    print("\n--- Cost Analysis @ 5,000 queries/day ---")
    gpt4o_input_per_1k   = 0.005   # $ per 1K input tokens (gpt-4o)
    mini_input_per_1k    = 0.00015 # $ per 1K input tokens (gpt-4o-mini)
    avg_tokens_per_query = 500     # input + output est.

    complex_fraction = sum(1 for r in rows_llm if r["model"] == MODEL_COMPLEX) / len(rows_llm)
    simple_fraction  = 1 - complex_fraction

    daily_queries = 5000
    cost_no_routing   = daily_queries * (avg_tokens_per_query / 1000) * gpt4o_input_per_1k
    cost_with_routing = (daily_queries * complex_fraction * (avg_tokens_per_query / 1000) * gpt4o_input_per_1k +
                         daily_queries * simple_fraction  * (avg_tokens_per_query / 1000) * mini_input_per_1k)
    saving = cost_no_routing - cost_with_routing

    print(f"  Complex fraction (LLM estimate): {complex_fraction:.0%}")
    print(f"  Cost without routing (all gpt-4o):    ${cost_no_routing:.2f}/day")
    print(f"  Cost with routing:                     ${cost_with_routing:.2f}/day")
    print(f"  Estimated daily saving:                ${saving:.2f}  (~{saving/cost_no_routing:.0%} reduction)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",    action="store_true", help="P1B.1 regex routing table (10 queries, no API)")
    parser.add_argument("--compare", action="store_true", help="P1B.2 regex vs LLM comparison (makes API calls)")
    args = parser.parse_args()

    if args.compare:
        _run_compare()
    else:
        _run_demo()
