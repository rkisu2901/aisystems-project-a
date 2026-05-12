"""
retry_demo.py — S4.2: Instructor retry loop visible.

Demonstrates instructor's automatic retry-on-validation-failure mechanism.
Uses StrictTicket — a deliberately narrow model with:
  - customer_sentiment restricted to 3 values (model often returns "confused", "upset", etc.)
  - summary min_length=40 (short/vague queries produce summaries that fail)
  - a custom validator that rejects summaries starting with "The customer"

The system prompt intentionally withholds the valid sentiment values so the model
must learn them from instructor's validation-error feedback on retries.

Call counting: patches OpenAI client's create() before wrapping with instructor so
every API call (first attempt + retries) is recorded.

Usage:
    python scripts/retry_demo.py
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from typing import Literal
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()

try:
    import instructor
    from openai import OpenAI
    INSTRUCTOR_ENABLED = True
except ImportError:
    print("ERROR: instructor and openai packages are required. pip install instructor openai")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Deliberately strict model — narrow constraints trigger retries
# ---------------------------------------------------------------------------

class StrictTicket(BaseModel):
    """Narrow response contract designed to expose instructor's retry loop.

    Differences from SupportTicket (which is permissive by design):
      - sentiment is Literal with only 3 values (not a free-text field)
      - summary has min_length=40 enforced by Pydantic Field
      - custom validator rejects summaries that start with "The customer"
        (a common LLM pattern that adds no info)
    """

    team: Literal["returns", "billing", "technical", "account", "general"]
    priority: Literal["low", "medium", "high", "urgent"]
    sentiment: Literal["angry", "frustrated", "neutral"] = Field(
        description="Customer emotional tone — exactly one of the allowed values."
    )
    summary: str = Field(
        min_length=40,
        description="One sentence (min 40 chars) describing the specific customer problem.",
    )

    @field_validator("summary")
    @classmethod
    def no_generic_opener(cls, v: str) -> str:
        if v.lower().startswith("the customer"):
            raise ValueError(
                "summary must be specific, not start with 'The customer …'. "
                "Describe the actual issue instead."
            )
        return v


# ---------------------------------------------------------------------------
# Call counter — patches client BEFORE instructor wraps it
# ---------------------------------------------------------------------------

# Using a list so the int is mutable from inside the closure.
_call_count = [0]

_raw_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_orig_create = _raw_client.chat.completions.create  # saved before patch — avoids infinite recursion


def _tracked_create(**kwargs):
    _call_count[0] += 1
    return _orig_create(**kwargs)


_raw_client.chat.completions.create = _tracked_create
_client = instructor.from_openai(_raw_client)


# ---------------------------------------------------------------------------
# Boundary-case queries (intentionally probe failure modes)
# ---------------------------------------------------------------------------

QUERIES = [
    (
        "Ambiguous sentiment",
        "I'm not sure if I should be upset or just confused — my statement shows "
        "a charge I don't recognise but it's a small amount.",
        # Model naturally returns "confused" or "uncertain" → fails 3-value enum → retry
    ),
    (
        "Vague/short query",
        "Help.",
        # Too little info → model generates a very short summary (<40 chars) → retry
    ),
    (
        "Multi-issue (team ambiguity)",
        "App keeps crashing whenever I try to view my EMI schedule, and I think "
        "I was charged twice last month as well.",
        # Model may toggle between 'technical' and 'billing' or return 'apologetic' sentiment
    ),
    (
        "Formal complaint (generic opener risk)",
        "I wish to formally raise a complaint regarding the delay in processing "
        "my account closure request submitted three weeks ago.",
        # Formal queries often produce "The customer wishes to …" summaries → validator rejects
    ),
    (
        "Polite customer (sentiment edge)",
        "Could you please help me understand why my loan prepayment penalty "
        "was calculated differently than what was shown in the app?",
        # Model may return "polite" or "confused" — only "neutral" is in enum for calm queries
    ),
]

SYSTEM_PROMPT = (
    "You are a customer support triage agent for Novus Bank. "
    "For the customer message below, produce a structured ticket. "
    "Be specific and concise.\n\n"
    "Customer message:\n<message>{query}</message>"
)


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def run_demo() -> None:
    print("=== S4.2 — Instructor Retry Loop Demo ===\n")
    print(
        f"{'#':<3} {'Label':<30} {'Attempts':>9} {'Outcome':<14} "
        f"{'Sentiment':<12} {'Time':>7}"
    )
    print("-" * 78)

    rows = []
    for i, (label, query, _) in enumerate(QUERIES, 1):
        _call_count[0] = 0  # reset per query
        t0 = time.time()
        try:
            ticket = _client.chat.completions.create(
                model="gpt-4o-mini",
                response_model=StrictTicket,
                max_retries=3,
                messages=[{"role": "user", "content": SYSTEM_PROMPT.format(query=query)}],
                temperature=0.4,  # higher than prod to increase variation → more observable retries
            )
            attempts = _call_count[0]
            outcome = "success"
            sentiment = ticket.sentiment
            summary_preview = ticket.summary[:60]
        except Exception as exc:
            attempts = _call_count[0]
            outcome = f"FAIL ({type(exc).__name__})"
            sentiment = "-"
            summary_preview = str(exc)[:60]

        elapsed = round(time.time() - t0, 2)
        rows.append(
            dict(
                num=i,
                label=label,
                attempts=attempts,
                outcome=outcome,
                sentiment=sentiment,
                elapsed=elapsed,
                summary_preview=summary_preview,
            )
        )
        print(
            f"{i:<3} {label:<30} {attempts:>9} {outcome:<14} "
            f"{sentiment:<12} {elapsed:>6.2f}s"
        )

    print("\n--- Summary preview (first 60 chars) ---")
    for r in rows:
        marker = "  " if r["attempts"] == 1 else "* "
        print(f"  {marker}#{r['num']} ({r['attempts']} attempt{'s' if r['attempts'] > 1 else ''}): {r['summary_preview']}")

    total_calls = sum(r["attempts"] for r in rows)
    retried = sum(1 for r in rows if r["attempts"] > 1)
    print(f"\nTotal API calls: {total_calls} across {len(rows)} queries")
    print(f"Queries that triggered retries: {retried}/{len(rows)}")

    print("""
--- Analysis: When does retry logic become a latency/cost problem? ---

At max_retries=3, each retry is a full LLM round-trip adding ~0.5–2s and
~200–400 input tokens. A query that exhausts all 3 retries costs 4x the base
call and takes 4–8s total — well above a 3s customer-support SLA (p95 target).
At 1,000 queries/day with a 10% retry rate, that is ~100 extra calls/day
(~$0.05/day at gpt-4o-mini pricing), but the latency tail matters far more
than the cost: a single 8s response in a chatbot triggers abandonment.

Alternative: Use OpenAI's native JSON Schema mode (response_format with a
strict JSON Schema) instead of Pydantic + instructor. The model validates
against the schema server-side before returning, so invalid structure is
impossible — eliminating the retry loop entirely. The trade-off: you lose
Pydantic's semantic validators (min_length, custom rules), but gain sub-1s
structured outputs with zero retries.
""")


if __name__ == "__main__":
    run_demo()
