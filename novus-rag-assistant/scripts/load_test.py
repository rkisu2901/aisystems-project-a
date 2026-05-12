"""
load_test.py — aiohttp-based load tester (ab-equivalent) for the Novus Bank RAG API.

Usage:
    python scripts/load_test.py --url URL --n 50 --c 1 [--payload PATH]
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp


async def _worker(session, url, payload, results, sem):
    async with sem:
        t0 = time.perf_counter()
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                await resp.read()
                elapsed = (time.perf_counter() - t0) * 1000  # ms
                results.append((resp.status, elapsed))
        except Exception:
            elapsed = (time.perf_counter() - t0) * 1000
            results.append((0, elapsed))


async def run(url, n, c, payload):
    sem = asyncio.Semaphore(c)
    results = []
    wall_start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [_worker(session, url, payload, results, sem) for _ in range(n)]
        await asyncio.gather(*tasks)
    wall = time.perf_counter() - wall_start

    statuses = [r[0] for r in results]
    latencies = sorted(r[1] for r in results)
    failed = sum(1 for s in statuses if s != 200)
    ok = len(latencies)

    def pct(p):
        idx = int(len(latencies) * p / 100)
        return round(latencies[min(idx, ok - 1)], 1)

    print(f"\n{'='*55}")
    print(f"  URL          : {url}")
    print(f"  Requests     : {n}  Concurrency: {c}")
    print(f"{'='*55}")
    print(f"  Complete     : {ok}")
    print(f"  Failed       : {failed}")
    print(f"  Total time   : {wall:.2f} s")
    print(f"  Req/sec      : {n/wall:.2f}")
    print(f"  Mean latency : {statistics.mean(latencies):.1f} ms")
    print(f"  Median (p50) : {pct(50)} ms")
    print(f"  p90          : {pct(90)} ms")
    print(f"  p95          : {pct(95)} ms")
    print(f"  p99          : {pct(99)} ms")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--c", type=int, default=1)
    parser.add_argument("--payload", default=None)
    args = parser.parse_args()

    if args.payload:
        payload = json.loads(Path(args.payload).read_text())
    else:
        payload = {"query": "What is the UPI daily transaction limit?", "mode": "dense"}

    asyncio.run(run(args.url, args.n, args.c, payload))


if __name__ == "__main__":
    main()
