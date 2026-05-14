"""
output_guardrail.py — O2.1: post-generation hallucination detector.

check_hallucination(answer, context) -> HallucinationResult
  Extracts every factual claim from the answer, verifies each against context,
  returns structured result with per-claim evidence and overall has_hallucination flag.

  Fails CLOSED on error: if the judge call fails, has_hallucination=True is returned
  so the pipeline falls back to a safe message rather than silently passing a
  potentially hallucinated answer. Appropriate for banking/finance domains.

FALLBACK_ANSWER: safe message returned by ask() when output guardrail fires twice.

Usage:
    from scripts.output_guardrail import check_hallucination, FALLBACK_ANSWER
    result = check_hallucination(answer, context)
    if result.has_hallucination:
        print("WARN:", [c.claim for c in result.claims if not c.supported])
"""

import os
import json
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

HALLUCINATION_PROMPT = """You are a factual accuracy checker for a banking AI assistant.

Given an ANSWER and the CONTEXT passages used to generate it:
1. Identify every factual claim in the ANSWER (numbers, percentages, policy details, timelines, names).
2. For each claim, check if it is directly supported by the CONTEXT.
3. Return a JSON object with this exact structure:
{
  "claims": [
    {"claim": "<the specific claim>", "supported": true/false, "evidence": "<quote from context OR 'not found'>"}
  ],
  "has_hallucination": true/false
}

Rules:
- "supported" = true ONLY if the claim is explicitly stated in context (not inferred).
- "has_hallucination" = true if ANY claim has "supported": false.
- Include only factual claims — skip filler phrases like "I hope this helps".
- Return ONLY the JSON object, no other text."""

# Shown to the customer when both generation attempts fail the output guardrail.
FALLBACK_ANSWER = (
    "I found some relevant information but cannot confirm all the details with "
    "full accuracy. Please contact Novus Bank support at 1800-NOVUS for a "
    "verified answer to your question."
)


@dataclass
class Claim:
    claim: str
    supported: bool
    evidence: str


@dataclass
class HallucinationResult:
    claims: list[Claim]
    has_hallucination: bool


def check_hallucination(answer: str, context: str) -> HallucinationResult:
    """Verify all factual claims in answer against context.

    Returns HallucinationResult with per-claim breakdown.
    Fails CLOSED on API error or parse error: returns has_hallucination=True
    so that a broken judge blocks the answer rather than silently passing it.
    """
    prompt = f"CONTEXT:\n{context[:4000]}\n\nANSWER:\n{answer}"
    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=600,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": HALLUCINATION_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        claims = [
            Claim(
                claim=c["claim"],
                supported=c["supported"],
                evidence=c.get("evidence", ""),
            )
            for c in data.get("claims", [])
        ]
        # Re-derive has_hallucination from claims to guard against model
        # inconsistency (e.g. has_hallucination=false but unsupported claims listed).
        has_hallucination = any(not c.supported for c in claims) or data.get("has_hallucination", False)
        return HallucinationResult(claims=claims, has_hallucination=has_hallucination)
    except Exception:
        # Fail closed: treat judge failure as a detected hallucination.
        return HallucinationResult(claims=[], has_hallucination=True)
