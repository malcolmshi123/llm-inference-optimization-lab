"""Compare benchmark metrics by workload at concurrency 8."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

def load_batches(path):
    """Read the top-level list of batches with their nested request results."""
    with path.open(encoding="utf-8") as handle:
        batches = json.load(handle)
    if not isinstance(batches, list):
        raise ValueError(f"{path}: expected a JSON list of batches")
    for index, batch in enumerate(batches):
        if not isinstance(batch, dict):
            raise ValueError(f"{path}: batch {index} must be an object")
        if not isinstance(batch.get("workload"), str):
            raise ValueError(f"{path}: batch {index} needs a workload string")
        if not isinstance(batch.get("results"), list):
            raise ValueError(f"{path}: batch {index} needs a results list")
    return batches

def group_by_workload(batches):
    """Keep complete batches, including trials and concurrency settings."""
    grouped = defaultdict(list)
    for batch in batches:
        grouped[(batch["workload"], batch["concurrency"])].append(batch)
    return dict(grouped)

def summarize_group(batches):
    """Summarize pooled successful requests; return None for unavailable metrics."""
    requests = [
        request
        for batch in batches
        for request in batch["results"]
        if request["success"]
    ]
    ttft_values = [
        request["ttft_ms"] for request in requests
        if request["ttft_ms"] is not None
    ]
    e2e_values = [
        request["e2e_ms"] for request in requests
        if request["e2e_ms"] is not None
    ]
    throughput_values = [
        batch["output_tokens_per_sec"] for batch in batches
        if batch["output_tokens_per_sec"] is not None
    ]

    return {
        "ttft_p50_ms": float(np.percentile(ttft_values, 50)) if ttft_values else None,
        "ttft_p95_ms": float(np.percentile(ttft_values, 95)) if ttft_values else None,
        "e2e_p50_ms": float(np.percentile(e2e_values, 50)) if e2e_values else None,
        "e2e_p95_ms": float(np.percentile(e2e_values, 95)) if e2e_values else None,
        "output_tokens_per_sec_mean": (
            statistics.mean(throughput_values) if throughput_values else None
        ),
    }

def compare_summaries(baseline, candidate):
    """Return signed percentage changes, or None for missing/zero baselines."""
    changes = {}
    for metric in dict.fromkeys([*baseline, *candidate]):
        baseline_value = baseline.get(metric)
        candidate_value = candidate.get(metric)
        if baseline_value is None or candidate_value is None or baseline_value == 0:
            changes[metric] = None
        else:
            changes[metric] = round(
                (candidate_value - baseline_value) / baseline_value * 100, 2
            )
    return changes

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Baseline benchmark JSON")
    parser.add_argument("candidate", type=Path, help="Candidate benchmark JSON")
    args = parser.parse_args()

    try:
        baseline = group_by_workload(load_batches(args.baseline))
        candidate = group_by_workload(load_batches(args.candidate))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Baseline: {args.baseline}")
    print(f"Candidate: {args.candidate}")
    groups = sorted(
        key for key in baseline.keys() | candidate.keys() if key[1] == 8
    )
    if not groups:
        print("No batches found at concurrency 8.")
        return

    for workload, concurrency in groups:
        baseline_batches = baseline.get((workload, concurrency), [])
        candidate_batches = candidate.get((workload, concurrency), [])
        baseline_summary = summarize_group(baseline_batches)
        candidate_summary = summarize_group(candidate_batches)
        changes = compare_summaries(baseline_summary, candidate_summary)

        print(f"\nWorkload: {workload}, concurrency: {concurrency}")
        print(
            f"Batches: baseline={len(baseline_batches)}, "
            f"candidate={len(candidate_batches)}"
        )
        print(f"{'Metric':<28} {'Baseline':>14} {'Candidate':>14} {'Change':>12}")
        for metric in baseline_summary:
            baseline_value = baseline_summary[metric]
            candidate_value = candidate_summary[metric]
            change = changes[metric]
            baseline_display = f"{baseline_value:.2f}" if baseline_value is not None else "N/A"
            candidate_display = f"{candidate_value:.2f}" if candidate_value is not None else "N/A"
            change_display = f"{change:+.2f}%" if change is not None else "N/A"
            print(
                f"{metric:<28} {baseline_display:>14} "
                f"{candidate_display:>14} {change_display:>12}"
            )

    print("\nPositive latency changes are worse; positive throughput changes are better.")

if __name__ == "__main__":
    main()
