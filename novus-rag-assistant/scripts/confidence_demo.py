"""
confidence_demo.py — O2.2 deliverable: 10-query confidence score demonstration.

Tests generate_with_confidence() directly with static context excerpts
so no pgvector / docker infrastructure is required.

Queries:
  1-5   Clearly answerable — context fully covers the question (expect HIGH/MEDIUM)
  6-10  Ambiguous or out-of-scope — context absent or tangential (expect LOW → handoff)

Usage:
    python scripts/confidence_demo.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.rag import generate_with_confidence, INSTRUCTOR_ENABLED
from scripts.response_models import HANDOFF_MESSAGE

# ---------------------------------------------------------------------------
# Static context excerpts (from novus-rag-assistant corpus)
# ---------------------------------------------------------------------------

MEMBERSHIP_CONTEXT = """
Novus Bank Membership Tiers:
- Standard: Basic benefits, 3% cashback on utilities, minimum AQB ₹10,000
- Plus: 5% cashback on dining and travel, AQB ₹25,000, free locker (small)
- Elite: 8% cashback on all spends, AQB ₹1,00,000, dedicated RM, airport lounge access
Upgrade: Maintain required AQB for 3 consecutive months to auto-upgrade.
Downgrade: Failure to maintain AQB for 2 consecutive months triggers downgrade.
"""

LOAN_CONTEXT = """
Personal Loan — Novus Bank:
Eligibility: CIBIL score >= 750; salaried or self-employed.
Loan amount: Rs 50,000 to Rs 25,00,000.
Tenure: 12 to 60 months.
Interest rate: 11% (Elite), 13% (Plus), 16-18% (Standard) p.a.
Processing fee: 1% of loan amount (min Rs 999).
Disbursement: Within 24 hours for pre-approved; 3-5 working days otherwise.
Prepayment: Allowed after 6 EMIs; 2% penalty on outstanding.
EMI bounce fee: Rs 500 per instance.
"""

UPI_CONTEXT = """
UPI and Digital Payments — Novus Bank:
UPI daily limit: Rs 1,00,000 per day (NPCI standard).
UPI per-transaction limit: Rs 1,00,000.
IMPS limit: Rs 5,00,000 per transaction.
NEFT: No per-transaction limit; processed in batches every 30 minutes.
RTGS: Minimum Rs 2,00,000; real-time settlement.
"""

FRAUD_CONTEXT = """
Fraud and Dispute Policy:
Report unauthorised transactions within 3 business days for zero liability.
Beyond 3 days: customer bears loss proportional to delay.
Call 1800-NOVUS (24x7) immediately on suspecting fraud.
Provisional credit: within 10 working days of dispute filing.
Final resolution: within 45 days.
Chargeback available for debit card transactions within 120 days.
"""

# Minimal / off-topic context — deliberately thin so LOW confidence triggers
MINIMAL_CONTEXT = "Novus Bank offers savings, loans, and membership products."

# ---------------------------------------------------------------------------
# Test queries
# ---------------------------------------------------------------------------

TEST_CASES = [
    # --- CLEARLY ANSWERABLE (1-5, expect HIGH or MEDIUM) ---
    {
        "id": "Q1", "label": "ANSWERABLE",
        "context": MEMBERSHIP_CONTEXT,
        "query":   "What AQB do I need to maintain for Elite membership?",
    },
    {
        "id": "Q2", "label": "ANSWERABLE",
        "context": LOAN_CONTEXT,
        "query":   "What is the personal loan interest rate for a Plus member?",
    },
    {
        "id": "Q3", "label": "ANSWERABLE",
        "context": UPI_CONTEXT,
        "query":   "What is the UPI daily transaction limit?",
    },
    {
        "id": "Q4", "label": "ANSWERABLE",
        "context": FRAUD_CONTEXT,
        "query":   "How long do I have to report an unauthorised transaction for zero liability?",
    },
    {
        "id": "Q5", "label": "ANSWERABLE",
        "context": LOAN_CONTEXT,
        "query":   "Can I prepay my personal loan and what is the penalty?",
    },
    # --- AMBIGUOUS / OUT-OF-SCOPE (6-10, expect LOW → handoff) ---
    {
        "id": "Q6", "label": "AMBIGUOUS",
        "context": MINIMAL_CONTEXT,
        "query":   "What is the interest rate on Novus Bank fixed deposits?",
    },
    {
        "id": "Q7", "label": "AMBIGUOUS",
        "context": MINIMAL_CONTEXT,
        "query":   "Does Novus Bank offer home loans, and what are the rates?",
    },
    {
        "id": "Q8", "label": "AMBIGUOUS",
        "context": MEMBERSHIP_CONTEXT,
        "query":   "What is the credit card reward points redemption policy?",
    },
    {
        "id": "Q9", "label": "AMBIGUOUS",
        "context": MINIMAL_CONTEXT,
        "query":   "How do I apply for an NRI savings account?",
    },
    {
        "id": "Q10", "label": "AMBIGUOUS",
        "context": LOAN_CONTEXT,
        "query":   "My loan was rejected. Who is the grievance officer I can escalate to?",
    },
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not INSTRUCTOR_ENABLED:
        print("ERROR: `instructor` package not installed. Run: pip install instructor")
        sys.exit(1)

    print("=== O2.2 — Confidence-gated answers (10 queries) ===\n")
    print(f"{'ID':<4}  {'LABEL':<12}  {'CONF':<8}  {'HANDOFF':<8}  REASONING")
    print("-" * 90)

    high_count   = 0
    medium_count = 0
    low_count    = 0
    handoff_count = 0

    for tc in TEST_CASES:
        answer, confidence, reasoning = generate_with_confidence(tc["query"], tc["context"])
        is_handoff = answer == HANDOFF_MESSAGE

        if confidence == "high":   high_count += 1
        if confidence == "medium": medium_count += 1
        if confidence == "low":    low_count += 1
        if is_handoff:             handoff_count += 1

        handoff_flag = "YES" if is_handoff else "no"
        print(f"{tc['id']:<4}  {tc['label']:<12}  {confidence:<8}  {handoff_flag:<8}  {reasoning[:55]}")

    print()
    print(f"HIGH: {high_count}  MEDIUM: {medium_count}  LOW: {low_count}  "
          f"(handoffs triggered: {handoff_count})")
    print()
    print("--- Sample LOW-confidence handoff message ---")
    print(HANDOFF_MESSAGE)


if __name__ == "__main__":
    main()
