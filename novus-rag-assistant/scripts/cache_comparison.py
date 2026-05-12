"""
cache_comparison.py — A6.3: compare p50/p95 latency with cache=False vs cache=True.

Uses repeated_queries.json (70% repeated, 30% novel) and calls ask() directly.
The semantic cache is a module-level singleton so it persists across calls within
the same process — cache=True runs build up the cache organically.
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

QUERIES_FILE = Path(__file__).parent.parent / "repeated_queries.json"


def pct(latencies, p):
    s = sorted(latencies)
    idx = int(len(s) * p / 100)
    return round(s[min(idx, len(s)-1)], 0)


def run_queries(use_cache: bool) -> list[float]:
    """Run all queries, return per-query latency in ms."""
    from scripts.rag import ask, _cache

    # Reset cache between runs
    _cache.reset()

    queries = json.loads(QUERIES_FILE.read_text())
    latencies = []
    cache_hits = 0
    for item in queries:
        t0 = time.perf_counter()
        result = ask(item["query"], use_cache=use_cache)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)
        if result["retrieval_mode"] == "cache_hit":
            cache_hits += 1
    print(f"  cache={'ON ' if use_cache else 'OFF'} | cache_hits={cache_hits}/{len(queries)} | "
          f"p50={pct(latencies,50):.0f}ms p95={pct(latencies,95):.0f}ms mean={statistics.mean(latencies):.0f}ms")
    return latencies, cache_hits


def main():
    queries = json.loads(QUERIES_FILE.read_text())
    n = len(queries)
    repeated = sum(1 for i, q in enumerate(queries) if any(q["query"] == queries[j]["query"] for j in range(i)))
    print(f"\nA6.3 — Semantic Cache Load Comparison")
    print(f"Queries: {n} total, ~{repeated} repeated ({repeated/n*100:.0f}% repeat rate)")
    print()

    print("Run 1 — cache=OFF (baseline):")
    lat_off, hits_off = run_queries(use_cache=False)

    print("Run 2 — cache=ON:")
    lat_on, hits_on = run_queries(use_cache=True)

    print()
    print(f"{'Metric':<15} {'cache=OFF':>12} {'cache=ON':>12} {'delta':>10}")
    print("-" * 52)
    print(f"{'p50 (ms)':<15} {pct(lat_off,50):>12.0f} {pct(lat_on,50):>12.0f} {pct(lat_on,50)-pct(lat_off,50):>+10.0f}")
    print(f"{'p95 (ms)':<15} {pct(lat_off,95):>12.0f} {pct(lat_on,95):>12.0f} {pct(lat_on,95)-pct(lat_off,95):>+10.0f}")
    print(f"{'mean (ms)':<15} {statistics.mean(lat_off):>12.0f} {statistics.mean(lat_on):>12.0f} {statistics.mean(lat_on)-statistics.mean(lat_off):>+10.0f}")
    print(f"{'cache_hits':<15} {hits_off:>12} {hits_on:>12}")

    # Cost calculation
    print()
    print("COST CALCULATION (gpt-4o-mini pricing, 2026):")
    print("  5,000 queries/day × 35% hit rate × 30 days = 52,500 saved LLM calls/month")
    input_tokens_saved  = 52_500 * 500    # ~500 input tokens per RAG call
    output_tokens_saved = 52_500 * 200    # ~200 output tokens per RAG call
    # gpt-4o-mini: $0.15/1M input, $0.60/1M output
    input_cost  = input_tokens_saved  * 0.15  / 1_000_000
    output_cost = output_tokens_saved * 0.60  / 1_000_000
    total_saved = input_cost + output_cost
    print(f"  Input tokens saved : {input_tokens_saved:,} × $0.15/1M = ${input_cost:.2f}")
    print(f"  Output tokens saved: {output_tokens_saved:,} × $0.60/1M = ${output_cost:.2f}")
    print(f"  Total monthly saving: ${total_saved:.2f}")
    print(f"  Also saves ~{hits_on/n*100:.0f}% of embedding API calls (text-embedding-3-small).")


if __name__ == "__main__":
    main()

