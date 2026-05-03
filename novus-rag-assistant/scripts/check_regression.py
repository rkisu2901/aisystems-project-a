"""
check_regression.py — Compare current eval results against a saved baseline.

Fails (exit code 1) if any metric drops by more than THRESHOLD (5 percentage
points) relative to the baseline. This is designed to run in CI after any
change to the RAG pipeline (chunking strategy, model, prompt, etc.).

All 1-5 LLM scores are normalized to 0-1 before comparison so thresholds
are consistent across metric types (retrieval vs LLM-judge).

Usage:
    python scripts/check_regression.py
    python scripts/check_regression.py --threshold 0.03   # tighter 3pp threshold
    python scripts/check_regression.py --baseline path/to/other_baseline.json
"""

import argparse
import json
import sys
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "eval_results.json"
BASELINE_PATH = Path(__file__).parent / "baseline_scores.json"

THRESHOLD = 0.05   # 5 percentage points — a meaningful signal, not noise

METRICS = ["hit_rate", "mrr", "faithfulness", "correctness"]
# Faithfulness and correctness are stored normalized (0-1) in the scorecard.
# hit_rate and mrr are already 0-1.


def load_scorecard(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    # eval_results.json wraps scorecard; baseline_scores.json has it at top level
    return data.get("scorecard", data)


def check(baseline: dict, current: dict, threshold: float) -> tuple[bool, list[str]]:
    """Compare current vs baseline. Return (passed, list_of_failures)."""
    failures = []

    for metric in METRICS:
        base_val = baseline.get(metric)
        curr_val = current.get(metric)

        if base_val is None or curr_val is None:
            print(f"  ⚠️  Metric '{metric}' missing in one of the files — skipping.")
            continue

        drop = base_val - curr_val
        status = "✅" if drop <= threshold else "❌"
        print(
            f"  {status}  {metric:<20} baseline={base_val:.4f}  "
            f"current={curr_val:.4f}  delta={-drop:+.4f}"
        )

        if drop > threshold:
            failures.append(
                f"{metric}: dropped {drop:.4f} (>{threshold:.4f} threshold). "
                f"Baseline={base_val:.4f}, Current={curr_val:.4f}"
            )

    return len(failures) == 0, failures


def main():
    parser = argparse.ArgumentParser(description="Novus Bank RAG regression checker")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--baseline", type=str, default=str(BASELINE_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_PATH))
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    results_path = Path(args.results)

    if not baseline_path.exists():
        print(f"No baseline found at {baseline_path}.")
        print("Run `python scripts/eval_harness.py --save-baseline` first.")
        sys.exit(2)

    if not results_path.exists():
        print(f"No eval results found at {results_path}.")
        print("Run `python scripts/eval_harness.py` first.")
        sys.exit(2)

    baseline = load_scorecard(baseline_path)
    current = load_scorecard(results_path)

    print(f"\nRegression check (threshold={args.threshold:.0%})")
    print(f"  Baseline : {baseline_path}")
    print(f"  Current  : {results_path}")
    print()

    passed, failures = check(baseline, current, args.threshold)

    print()
    if passed:
        print("✅  NO REGRESSION DETECTED — all metrics within threshold.")
        sys.exit(0)
    else:
        print("❌  REGRESSION DETECTED:")
        for f in failures:
            print(f"    • {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
