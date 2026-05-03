"""
synthetic_generator.py — Generate synthetic Q&A pairs from Novus Bank policy docs.

Three personas model the diversity of real customer questions:
  - standard    : clear, direct, neutral phrasing
  - frustrated  : emotional, mentions prior failed attempts
  - mismatch    : asks about one product but conflates it with another
                  (e.g., treats FD interest like loan interest)

The --critique flag enables an auto-critique loop: GPT-4o-mini scores each
generated pair for alignment between the question and expected_answer.
Pairs scoring below CRITIQUE_THRESHOLD are dropped, reducing hallucinated
or ambiguous training data before it enters the golden dataset.

Usage:
    python scripts/synthetic_generator.py --doc 02_membership_tiers.md
    python scripts/synthetic_generator.py --doc 07_emi_and_repayment.md --persona frustrated
    python scripts/synthetic_generator.py --doc 01_account_opening_policy.md --critique
    python scripts/synthetic_generator.py --all --critique --output my_extra_pairs.json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GEN_MODEL = "gpt-4o-mini"
CRITIQUE_MODEL = "gpt-4o-mini"
CRITIQUE_THRESHOLD = 3   # drop pairs scoring below this (1-5 scale)
GEN_TEMPERATURE = 0.8    # higher temp for creative question variety


# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------

PERSONA_PROMPTS = {
    "standard": (
        "You are a Novus Bank customer with a clear, specific question about the policy. "
        "Generate direct, neutral questions a new customer might ask after reading an FAQ."
    ),
    "frustrated": (
        "You are a Novus Bank customer who has already tried to get help and failed. "
        "Your questions are slightly emotional and reference a past problem. "
        "E.g., 'I've been waiting 5 days and nobody told me...' or 'Why wasn't I warned about...'"
    ),
    "mismatch": (
        "You are a Novus Bank customer who confuses two similar products or policies. "
        "For example, you might ask about FD penalty using loan prepayment terminology, "
        "or ask about credit card interest using savings account terms. "
        "Generate questions that contain a plausible but incorrect assumption."
    ),
}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

GENERATION_PROMPT = """You are building a golden evaluation dataset for a banking RAG system.

Persona: {persona_description}

Below is an excerpt from a Novus Bank policy document (doc_id: {doc_id}).
Generate {n_pairs} question-answer pairs that a Novus Bank customer might ask
about the content in this excerpt.

Rules:
- Each question must be answerable using ONLY the provided text.
- The expected_answer must be a precise, complete answer drawn from the text.
- Assign difficulty: "easy" (single fact lookup), "medium" (requires combining 2 facts),
  or "hard" (requires reasoning across multiple conditions or edge cases).
- Assign a category from: account, payments, loans, products, fraud, membership, grievance.

Document excerpt:
{content}

Respond with valid JSON only — a list of objects, each with:
  "query", "expected_answer", "difficulty", "category"

No other text outside the JSON.
"""

CRITIQUE_PROMPT = """You are a quality reviewer for a banking Q&A dataset.

Rate this question-answer pair for ALIGNMENT: does the expected_answer directly
and completely answer the question using only the document?

Question: {query}
Expected Answer: {expected_answer}
Document excerpt: {content}

Scoring:
  5 — Perfect alignment; answer is accurate and complete.
  4 — Good; minor detail could be added.
  3 — Acceptable; answer is correct but vague or incomplete.
  2 — Weak; partial answer or question is confusingly worded.
  1 — Poor; answer doesn't address the question or contains errors.

Respond with valid JSON only:
{{"score": <1-5>, "reason": "<one sentence>"}}
"""


def load_document(doc_path: Path) -> str:
    return doc_path.read_text(encoding="utf-8")


def generate_pairs(doc_id: str, content: str, persona: str, n_pairs: int = 5) -> list[dict]:
    """Call GPT to generate n_pairs Q&A pairs from a doc excerpt."""
    persona_desc = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["standard"])
    # Truncate content to avoid token overflow for large docs
    excerpt = content[:3000]

    prompt = GENERATION_PROMPT.format(
        persona_description=persona_desc,
        doc_id=doc_id,
        n_pairs=n_pairs,
        content=excerpt,
    )

    response = client.chat.completions.create(
        model=GEN_MODEL,
        temperature=GEN_TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    parsed = json.loads(raw)

    # Handle both {"pairs": [...]} and direct list responses
    if isinstance(parsed, list):
        pairs = parsed
    elif isinstance(parsed, dict):
        pairs = next(iter(parsed.values()))
    else:
        pairs = []

    # Inject metadata
    for i, pair in enumerate(pairs):
        pair["id"] = f"{doc_id.upper()[:8]}-SYN-{persona[:3].upper()}-{i+1:02d}"
        pair["expected_source"] = doc_id
        pair["persona"] = persona

    return pairs


def critique_pairs(pairs: list[dict], content: str, threshold: int = CRITIQUE_THRESHOLD) -> tuple[list[dict], list[dict]]:
    """Score each pair and split into kept (score >= threshold) and dropped."""
    kept = []
    dropped = []

    for pair in pairs:
        prompt = CRITIQUE_PROMPT.format(
            query=pair["query"],
            expected_answer=pair["expected_answer"],
            content=content[:2000],
        )
        response = client.chat.completions.create(
            model=CRITIQUE_MODEL,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        pair["critique_score"] = result.get("score", 0)
        pair["critique_reason"] = result.get("reason", "")

        if pair["critique_score"] >= threshold:
            kept.append(pair)
        else:
            dropped.append(pair)

        time.sleep(0.1)  # avoid rate limit burst

    return kept, dropped


def process_document(
    doc_path: Path,
    persona: str,
    n_pairs: int,
    use_critique: bool,
) -> list[dict]:
    doc_id = doc_path.stem
    content = load_document(doc_path)

    print(f"  Generating {n_pairs} pairs from {doc_id} (persona={persona})…")
    pairs = generate_pairs(doc_id, content, persona, n_pairs)

    if use_critique:
        kept, dropped = critique_pairs(pairs, content)
        drop_rate = len(dropped) / max(len(pairs), 1)
        print(f"    Auto-critique: kept {len(kept)}/{len(pairs)} pairs (drop rate={drop_rate:.0%})")
        return kept
    else:
        return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Synthetic Q&A generator for Novus Bank")
    parser.add_argument("--doc", type=str, help="Single document filename (e.g. 02_membership_tiers.md)")
    parser.add_argument("--all", action="store_true", help="Process all 19 corpus documents")
    parser.add_argument(
        "--persona",
        choices=["standard", "frustrated", "mismatch", "all"],
        default="standard",
        help="Customer persona to use (default: standard)",
    )
    parser.add_argument("--n-pairs", type=int, default=5, help="Pairs per doc per persona (default: 5)")
    parser.add_argument("--critique", action="store_true", help="Enable auto-critique loop")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    # Determine which documents to process
    if args.all:
        doc_paths = sorted(CORPUS_DIR.glob("*.md"))
    elif args.doc:
        doc_paths = [CORPUS_DIR / args.doc]
        if not doc_paths[0].exists():
            print(f"Error: {doc_paths[0]} not found.")
            sys.exit(1)
    else:
        print("Specify --doc <filename> or --all")
        sys.exit(1)

    # Determine which personas to use
    personas = list(PERSONA_PROMPTS.keys()) if args.persona == "all" else [args.persona]

    all_pairs = []
    for doc_path in doc_paths:
        for persona in personas:
            pairs = process_document(doc_path, persona, args.n_pairs, args.critique)
            all_pairs.extend(pairs)
            time.sleep(0.2)

    # Display results
    print(f"\nGenerated {len(all_pairs)} synthetic Q&A pairs\n")
    for pair in all_pairs:
        print(f"  [{pair['id']}] ({pair['difficulty']}/{pair['category']})")
        print(f"    Q: {pair['query']}")
        print(f"    A: {pair['expected_answer'][:100]}…")
        if "critique_score" in pair:
            print(f"    Critique: {pair['critique_score']}/5 — {pair['critique_reason']}")
        print()

    # Save if requested
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = None

    if output_path:
        output_path.write_text(json.dumps(all_pairs, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved → {output_path}")

    return all_pairs


if __name__ == "__main__":
    main()
