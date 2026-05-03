# Failure Taxonomy — Novus Bank RAG (Week 1 Baseline)

## Overview

The Week 1 baseline pipeline uses fixed-size chunking (500 chars, no overlap)
with cosine retrieval (top-5) and GPT-4o-mini generation. This document
catalogs the 5 failure modes most likely to appear in the baseline eval run.

Understanding these failures is the starting point for Week 2 improvements.
Each failure type has a root cause and a proposed fix.

---

## Failure Type 1: Retrieval Miss — Boundary Split

**What happens:** The answer to a query spans two adjacent 500-char chunks.
The retriever surfaces both halves, but neither contains enough context alone
for the LLM to generate a correct answer.

**Example queries likely to fail:**
- "What is the effective rate if I break a 2-year FD at 14 months?"
  → The penalty calculation logic spans the example sentence that straddles
    two chunks in `10_fixed_deposit_policy`.
- "What is the NPS Tier 2 minimum contribution and lock-in period?"
  → Tier 1 and Tier 2 details are in adjacent chunks; query mentions both.

**Metric impact:** `hit=True` (correct doc is retrieved) but `correctness ≤ 3`
because the LLM sees only half the needed information.

**Root cause:** No chunk overlap means boundary sentences are only in one chunk.

**Week 2 fix:** Switch to 200-char overlap or sentence-aware chunking.

---

## Failure Type 2: Wrong Ranking — Vocabulary Overlap

**What happens:** A query's wording matches a less-relevant document more
strongly than the target document, pushing the correct doc out of top-3.

**Example queries likely to fail:**
- "What is the grace period for loan EMI payment?"
  → "grace period" appears prominently in `07_emi_and_repayment` (correct)
    but also in `15_account_closure_and_dormancy` (re-KYC grace period).
    The wrong doc may rank higher if the EMI chunk has less overlap.
- "What happens if I miss the KYC deadline?"
  → `13_kyc_and_compliance` and `15_account_closure_and_dormancy` both
    discuss this; the dormancy doc may surface first.

**Metric impact:** `hit=True` but `mrr < 0.5` (correct doc at rank 3 or 4);
context assembly includes noise that confuses the LLM.

**Root cause:** text-embedding-3-small embeds surface vocabulary, not intent.

**Week 2 fix:** HyDE (hypothetical document embeddings) or query expansion
to reduce vocabulary sensitivity.

---

## Failure Type 3: Hallucination — Partial Context + High Confidence

**What happens:** The retrieved context contains a related but incomplete fact.
GPT-4o-mini fills in the missing detail from its training data rather than
saying "I don't have that information."

**Example queries likely to fail:**
- "What is the interest rate for a 3-year NPS-linked FD?"
  → NPS and FD are separate documents. The LLM may synthesize a plausible
    but fabricated rate by combining knowledge of both.
- "Can I use my foreign exchange card to withdraw from Novus Bank ATMs?"
  → Forex card policy mentions "Novus Bank ATMs" but the domestic ATM
    policy and forex card policy are in different chunks/docs.

**Metric impact:** `faithfulness ≤ 2` — the answer adds information not in
the retrieved context.

**Root cause:** GPT-4o-mini's system prompt says "only from context" but at
temperature=0.1 it occasionally interpolates. Also: two-hop queries where
both hops need separate retrievals.

**Week 2 fix:** Stricter system prompt wording; add a "NOT IN CONTEXT" guard
phrase to the prompt; consider re-ranking to ensure both hops are present.

---

## Failure Type 4: Wrong Interpretation — Tier Conflation

**What happens:** A query asks about a specific membership tier, but the
retrieved chunk describes a different tier's rules. The LLM applies the wrong
tier's numbers.

**Example queries likely to fail:**
- "How many free NEFT transactions does a Plus customer get?"
  → Chunks from `09_neft_rtgs_imps` and `02_membership_tiers` both mention
    NEFT limits; if the Standard-tier row is retrieved alongside the Plus
    row, the LLM may average them or pick the wrong one.
- "What is the personal loan interest rate for Elite customers?"
  → The rate table in `05_personal_loan_policy` spans multiple chunks.
    The Elite row may be in a different chunk than the table header.

**Metric impact:** `correctness = 2-3` — the answer addresses the right topic
but with wrong numbers for the tier.

**Root cause:** Tabular data loses row-column associations when chunked by
character count. The header row and data row may be split across chunks.

**Week 2 fix:** Table-aware chunking: keep full markdown table sections
together, or pre-extract tables as structured JSON metadata.

---

## Failure Type 5: Irrelevant Retrieval — Escalation / Edge Cases

**What happens:** Queries about specific edge cases (e.g., "I shared my OTP,
am I still covered?") retrieve surface-similar chunks from the wrong policy.

**Example queries likely to fail:**
- FRAUD-003: "I shared my OTP with someone and got defrauded. Am I covered?"
  → "OTP" appears in `08_upi_and_digital_payments` (UPI PIN security),
    `18_net_banking_and_app` (2FA), and `12_fraud_and_dispute_policy`
    (the correct one). Top-5 may not include the fraud liability section.
- KYC-002: "What happens if I don't complete re-KYC within the grace period?"
  → "grace period" matches dormancy policy as strongly as KYC policy.

**Metric impact:** `hit=False` for hard/edge-case queries → `mrr=0`.

**Root cause:** The correct chunk is a third or fourth cosine-distance match
because the query's discriminating phrase ("OTP + fraud" vs "OTP + UPI")
doesn't uniquely point to the fraud document at chunk level.

**Week 2 fix:** Metadata filtering — tag chunks with policy domain at ingest
time and apply a pre-filter before cosine search.

---

## Summary Table

| # | Type | Hit? | MRR | Faith | Correct | Week 2 Fix |
|---|------|------|-----|-------|---------|------------|
| 1 | Boundary split | ✅ | ≥0.5 | 4 | 2-3 | Overlapping chunks |
| 2 | Wrong ranking | ✅ | <0.5 | 4 | 3 | HyDE / query expansion |
| 3 | Hallucination | ✅ | ≥0.5 | ≤2 | 3 | Stricter prompt guard |
| 4 | Tier conflation | ✅ | ≥0.5 | 4 | 2 | Table-aware chunking |
| 5 | Irrelevant retrieval | ❌ | 0 | — | 1 | Metadata + pre-filter |

The **top 3 hardest categories** for the Week 1 baseline are expected to be:
`fraud`, `products` (FD/NPS edge cases), and `grievance` (multi-level rules).
These should show the largest correctness gains after Week 2 improvements.
