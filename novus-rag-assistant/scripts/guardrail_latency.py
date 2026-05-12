"""
guardrail_latency.py — G1.3: measure guardrail overhead for 20 queries.

For each query measures:
  (a) guard_ms    — check_input() alone
  (b) with_ms     — full ask() including check_input()
  (c) without_ms  — full ask() without any guardrail

Prints a 20-row table + summary + cache-threshold recommendation.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.input_guardrail import check_input
from scripts.rag import ask

QUERIES = [
    # 10 simple
    ("What is the UPI daily transaction limit?",           "simple"),
    ("How do I reset my net banking password?",            "simple"),
    ("What documents are needed for KYC?",                 "simple"),
    ("What is the minimum balance for a savings account?", "simple"),
    ("How long does an NEFT transfer take?",               "simple"),
    ("What is the interest rate on a fixed deposit?",      "simple"),
    ("Can I change my registered mobile number online?",   "simple"),
    ("What is the credit card annual fee?",                "simple"),
    ("How do I block my debit card?",                      "simple"),
    ("What is the cash withdrawal limit at ATMs?",         "simple"),
    # 10 complex
    ("If I do a UPI transfer of ₹5 lakh in two instalments will it be flagged?", "complex"),
    ("What happens to my loan EMI if I lose my job and miss three payments?",     "complex"),
    ("Compare the interest rates on home loans vs personal loans at Novus Bank.", "complex"),
    ("What are the tax implications of premature FD closure under Section 80C?",  "complex"),
    ("How does Novus Bank handle cross-border transactions in multiple currencies?","complex"),
    ("What is the dispute resolution process for a failed UPI payment?",          "complex"),
    ("Explain the difference between a lien and a pledge in the context of loans.","complex"),
    ("What compliance steps must I take for importing goods using a Letter of Credit?", "complex"),
    ("Under what conditions can the bank freeze my account without prior notice?", "complex"),
    ("How do I set up auto-pay for credit card dues linked to my savings account?","complex"),
]


def time_guardrail(query: str) -> float:
    t0 = time.perf_counter()
    check_input(query)
    return (time.perf_counter() - t0) * 1000


def time_full_with(query: str) -> float:
    t0 = time.perf_counter()
    check_input(query)
    ask(query)
    return (time.perf_counter() - t0) * 1000


def time_full_without(query: str) -> float:
    t0 = time.perf_counter()
    ask(query)
    return (time.perf_counter() - t0) * 1000


def main():
    print(f"\n{'Query':<52} {'Type':<8} {'guard_ms':>9} {'with_ms':>9} {'without_ms':>11} {'guard_pct':>10}")
    print("-" * 105)

    rows = []
    for query, qtype in QUERIES:
        guard_ms   = time_guardrail(query)
        with_ms    = time_full_with(query)
        without_ms = time_full_without(query)
        pct = round(guard_ms / with_ms * 100, 1) if with_ms > 0 else 0
        rows.append((query, qtype, guard_ms, with_ms, without_ms, pct))
        label = query[:50] + ".." if len(query) > 50 else query
        print(f"{label:<52} {qtype:<8} {guard_ms:>9.0f} {with_ms:>9.0f} {without_ms:>11.0f} {pct:>9.1f}%")

    # Summary
    simple  = [r for r in rows if r[1] == "simple"]
    complex_ = [r for r in rows if r[1] == "complex"]
    all_guard    = [r[2] for r in rows]
    all_with     = [r[3] for r in rows]
    all_without  = [r[4] for r in rows]
    all_pct      = [r[5] for r in rows]

    print("-" * 105)
    print(f"\n{'SUMMARY':}")
    print(f"  Mean guard_ms          : {sum(all_guard)/len(all_guard):.0f} ms")
    print(f"  Mean with_ms           : {sum(all_with)/len(all_with):.0f} ms")
    print(f"  Mean without_ms        : {sum(all_without)/len(all_without):.0f} ms")
    print(f"  Mean guard_pct         : {sum(all_pct)/len(all_pct):.1f}%")
    print(f"  Simple   guard_ms mean : {sum(r[2] for r in simple)/len(simple):.0f} ms")
    print(f"  Complex  guard_ms mean : {sum(r[2] for r in complex_)/len(complex_):.0f} ms")
    print()
    print("CACHE-THRESHOLD RECOMMENDATION:")
    mean_guard = sum(all_guard)/len(all_guard)
    mean_without = sum(all_without)/len(all_without)
    threshold_pct = mean_guard / mean_without * 100
    print(f"  Guardrail adds ~{mean_guard:.0f} ms to each request ({threshold_pct:.1f}% of pipeline without guard).")
    print(f"  Cache guardrail results when: query repeat rate > 30% AND guard_ms > 200 ms.")
    print(f"  Use SemanticCache (threshold=0.95) keyed on the raw query; TTL = 15 min.")
    print(f"  At {threshold_pct:.0f}% overhead, caching saves meaningful latency only at high traffic (>500 req/hr).")


if __name__ == "__main__":
    main()
