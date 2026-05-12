"""
response_models.py — Pydantic response contracts for structured LLM outputs.

Week 4 additions:
  O2.2: AnswerWithConfidence — confidence-gated answer generation
  S4.1: SupportTicket        — structured escalation ticket + generate_ticket()

All models are used via the `instructor` library which patches the OpenAI client
to validate and retry structured outputs automatically.

Usage:
    from scripts.response_models import AnswerWithConfidence, Confidence, HANDOFF_MESSAGE
    from scripts.response_models import SupportTicket, generate_ticket

    ticket = generate_ticket(query, context)
    print(ticket.model_dump_json(indent=2))

CLI:
    python scripts/response_models.py           # 3 built-in escalation test queries
    python scripts/response_models.py --query "..."
"""

import os
import sys
from enum import Enum
from typing import Literal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pydantic import BaseModel, Field

try:
    import instructor
    from openai import OpenAI as _OpenAI
    _raw_client = _OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    _instructor_client = instructor.from_openai(_raw_client)
    INSTRUCTOR_ENABLED = True
except ImportError:
    INSTRUCTOR_ENABLED = False


# ---------------------------------------------------------------------------
# O2.2 — Confidence-gated answers
# ---------------------------------------------------------------------------

class Confidence(str, Enum):
    HIGH   = "high"    # Every claim explicitly stated in the retrieved context
    MEDIUM = "medium"  # Most claims in context; minor inference or paraphrase
    LOW    = "low"     # Context only tangentially related; significant inference needed


class AnswerWithConfidence(BaseModel):
    answer:     str = Field(description="The answer to the customer's question")
    confidence: Confidence = Field(
        description=(
            "HIGH if every factual claim is explicitly in the context. "
            "MEDIUM if most claims are in context with minor inference. "
            "LOW if the context only tangentially relates or the question is out of scope."
        )
    )
    reasoning: str = Field(
        description="One sentence explaining why this confidence level was assigned."
    )


HANDOFF_MESSAGE = (
    "I want to give you accurate information, but I don't have enough context "
    "to answer this confidently. Please contact our support team at "
    "support@novusbank.com for accurate help."
)


# ---------------------------------------------------------------------------
# S4.1 — Structured escalation ticket
# ---------------------------------------------------------------------------

class SupportTicket(BaseModel):
    """Structured escalation ticket generated when the pipeline cannot answer confidently.

    Generated via instructor so every field is validated and retried automatically.
    Never returned for queries the pipeline handles normally — only for genuine escalations.
    """
    summary: str = Field(
        description=(
            "One-sentence summary of the customer's problem. "
            "Include the core issue and any relevant identifiers (order IDs, amounts)."
        )
    )
    team: Literal["returns", "billing", "technical", "account", "general"] = Field(
        description=(
            "Team best suited to handle this ticket. "
            "returns: refund/return/delivery disputes. "
            "billing: wrong charges, payment failures, EMI issues. "
            "technical: app/website errors, login problems. "
            "account: KYC, account freeze, verification. "
            "general: everything else."
        )
    )
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        description=(
            "Ticket priority. "
            "urgent: fraud, account compromise, large amount dispute (>₹10,000). "
            "high: billing error, delayed refund >7 days, angry customer. "
            "medium: standard query not in knowledge base, first-time complaint. "
            "low: general information request, minor inconvenience."
        )
    )
    customer_sentiment: str = Field(
        description=(
            "One adjective or short phrase describing the customer's emotional tone. "
            "Examples: frustrated, confused, angry, anxious, neutral, polite."
        )
    )
    what_was_tried: str = Field(
        description=(
            "What the automated system attempted before escalating. "
            "Be specific: which intent was classified, which tool was used, "
            "whether context was found, and why it was insufficient."
        )
    )
    suggested_action: str = Field(
        description=(
            "Concrete next step for the human agent. "
            "Examples: 'Verify order ORD-445521 status in OMS and issue manual refund if >7 days', "
            "'Check billing statement for double-charge on account and reverse penalty fee'."
        )
    )
    context_summary: str = Field(
        description=(
            "Brief summary (1–2 sentences) of the product knowledge that was retrieved, "
            "or 'No relevant context found' if retrieval returned nothing useful."
        )
    )


TICKET_SYSTEM_PROMPT = """You are a senior customer support triage specialist at Novus Bank.

A customer query has been escalated because the automated system could not answer it confidently.
Your job is to create a structured support ticket that helps a human agent resolve the issue quickly.

Customer query:
<query>{query}</query>

Retrieved product knowledge (may be empty or insufficient):
<context>{context}</context>

Fill every field accurately based on the query and context above.
Be specific and actionable — the human agent should be able to act immediately.
"""


def generate_ticket(
    query:   str,
    context: str = "",
    model:   str = "gpt-4o-mini",
) -> SupportTicket:
    """Generate a structured SupportTicket for a query that cannot be answered automatically.

    Uses instructor for validated structured output with automatic retry on validation failure.
    Falls back to a best-effort SupportTicket via plain JSON parse if instructor is unavailable.

    Args:
        query:   The original customer query (already PII-restored if anonymizer was used).
        context: Retrieved product knowledge (may be empty for out-of-corpus queries).
        model:   Chat model to use (default: gpt-4o-mini).

    Returns:
        SupportTicket with all 7 fields populated.
    """
    prompt = TICKET_SYSTEM_PROMPT.format(
        query=query,
        context=context[:3000] if context else "No relevant context retrieved.",
    )

    if INSTRUCTOR_ENABLED:
        ticket = _instructor_client.chat.completions.create(
            model=model,
            response_model=SupportTicket,
            max_retries=3,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return ticket

    # Graceful fallback if instructor not installed
    from openai import OpenAI as _OpenAI2
    _fb_client = _OpenAI2(api_key=os.getenv("OPENAI_API_KEY"))
    import json as _json
    raw = _fb_client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You must respond ONLY with valid JSON matching this exact schema "
                    "(no markdown, no explanation): "
                    '{"summary": str, "team": "returns|billing|technical|account|general", '
                    '"priority": "low|medium|high|urgent", "customer_sentiment": str, '
                    '"what_was_tried": str, "suggested_action": str, "context_summary": str}'
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    data = _json.loads(raw.choices[0].message.content.strip())
    return SupportTicket(**data)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

_ESCALATION_TEST_QUERIES = [
    (
        "Billing dispute",
        "I was charged ₹2,500 twice for my loan EMI in March — once on the 5th and again "
        "on the 12th. I've already paid and the second charge has left my account in negative. "
        "This is completely unacceptable and I want an immediate reversal.",
        "EMI deductions are processed on the due date. Duplicate EMI deductions are rare "
        "and must be investigated manually. Standard refund timeline is 5–7 business days.",
    ),
    (
        "Angry return demand",
        "I ordered a premium debit card (ORD-778899) 3 weeks ago and it still hasn't arrived. "
        "Your website says 7 days. I've called twice and no one can tell me anything. "
        "I want a refund and I want to close my account.",
        "Debit card delivery typically takes 5–7 business days. Delays beyond 14 days "
        "should be escalated to the card dispatch team. Account closure requests require "
        "branch visit with KYC documents.",
    ),
    (
        "Out-of-corpus query",
        "I want to know if Novus Bank participates in the RBI's Positive Pay System for "
        "cheques above ₹50,000 and whether I can register via the mobile app or only "
        "through net banking.",
        "",   # No relevant context — out of corpus
    ),
]


def main() -> None:
    import argparse, time
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="S4.1 — SupportTicket demo")
    parser.add_argument("--query", type=str, help="Single query to escalate")
    args = parser.parse_args()

    if args.query:
        queries = [("Custom", args.query, "")]
    else:
        queries = _ESCALATION_TEST_QUERIES

    print("=== S4.1 — SupportTicket structured escalation ===\n")
    for label, query, context in queries:
        print(f"[{label}]")
        print(f"Query   : {query[:120]}")
        t0 = time.time()
        ticket = generate_ticket(query, context)
        elapsed = round(time.time() - t0, 2)
        print(f"Summary : {ticket.summary}")
        print(f"Team    : {ticket.team}  |  Priority : {ticket.priority}")
        print(f"Sentiment     : {ticket.customer_sentiment}")
        print(f"What tried    : {ticket.what_was_tried}")
        print(f"Suggested act : {ticket.suggested_action}")
        print(f"Context summ  : {ticket.context_summary}")
        print(f"Time    : {elapsed}s")
        print("-" * 70)


if __name__ == "__main__":
    main()
