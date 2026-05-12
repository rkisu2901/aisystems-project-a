"""
output_guardrail.py — O2.1: post-generation hallucination detector.

check_hallucination(answer, context) -> HallucinationResult
  Extracts every factual claim from the answer, verifies each against context,
  returns structured result with per-claim evidence and overall has_hallucination flag.

Usage:
    from scripts.output_guardrail import check_hallucination
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
    Fails open (has_hallucination=False) if the LLM call fails.
    """
    prompt = f"CONTEXT:\n{context[:3000]}\n\nANSWER:\n{answer}"
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
        return HallucinationResult(
            claims=claims,
            has_hallucination=data.get("has_hallucination", False),
        )
    except Exception:
        return HallucinationResult(claims=[], has_hallucination=False)
