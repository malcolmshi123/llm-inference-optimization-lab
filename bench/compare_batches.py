import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

def validate_number(value, field, context, allow_none=False):
    if value is None and allow_none:
        return
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{context}: invalid {field} value: {value!r}")

def load_batches(path):
    with path.open(encoding="utf-8") as handle:
        batches = json.load(handle)
    if not isinstance(batches, list):
        raise ValueError(f"{path}: expected a JSON list")
    for index, batch in enumerate(batches):
        context = f"{path}: batch {index}"
        if not isinstance(batch, dict):
            raise ValueError(f"{context}: expected an object")
        if not isinstance(batch.get("workload"), str):
            raise ValueError(f"{context}: invalid workload")
        for field in ("concurrency", "target_tokens", "trial"):
            value = batch.get(field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{context}: invalid {field}")
        if "output_tokens_per_sec" not in batch:
            raise ValueError(f"{context}: missing output_tokens_per_sec")
        validate_number(
            batch["output_tokens_per_sec"],
            "output_tokens_per_sec",
            context,
            allow_none=True,
        )
        if "max_num_batched_tokens" in batch:
            budget = batch["max_num_batched_tokens"]
            if budget is not None and (type(budget) is not int or budget <= 0):
                raise ValueError(f"{context}: invalid max_num_batched_tokens")

        requests = batch.get("results")
        if not isinstance(requests, list):
            raise ValueError(f"{context}: invalid results list")
        for request_index, request in enumerate(requests):
            request_context = f"{context}: request {request_index}"
            if not isinstance(request, dict):
                raise ValueError(f"{request_context}: expected an object")
            if type(request.get("success")) is not bool:
                raise ValueError(f"{request_context}: invalid success")
            for field in ("ttft_ms", "e2e_ms"):
                if field not in request:
                    raise ValueError(f"{request_context}: missing {field}")
                validate_number(request[field], field, request_context, allow_none=True)

    if any("max_num_batched_tokens" in batch for batch in batches):
        if not all("max_num_batched_tokens" in batch for batch in batches):
            raise ValueError(f"{path}: missing max_num_batched_tokens in some batches")
        budgets = {batch["max_num_batched_tokens"] for batch in batches}
        if len(budgets) != 1:
            raise ValueError(f"{path}: Mixed scheduling budgets in benchmark file")
    return batches

def group_by_workload(batches):
    grouped = defaultdict(list)
    for batch in batches:
        key = (batch["workload"], batch["concurrency"])
        grouped[key].append(batch)
    for key, group in grouped.items():
        target_tokens = {batch["target_tokens"] for batch in group}
        if len(target_tokens) != 1:
            raise ValueError(
                f"{key}: mixed target_tokens values: {sorted(target_tokens)}"
            )
    return dict(grouped)

def summarize_group(batches):
    requests = [
        request
        for batch in batches
        for request in batch["results"]
        if request["success"]
    ]
    ttft_values = [
        request["ttft_ms"] for request in requests if request["ttft_ms"] is not None
    ]
    e2e_values = [
        request["e2e_ms"] for request in requests if request["e2e_ms"] is not None
    ]
    throughput_values = [
        batch["output_tokens_per_sec"]
        for batch in batches
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

def get_trial_ttfts(batches):
    trial_ttfts = defaultdict(list)
    for batch in batches:
        for request in batch["results"]:
            ttft = request.get("ttft_ms")
            if request["success"] and ttft is not None:
                trial_ttfts[batch["trial"]].append(ttft)
    return {trial: sorted(values) for trial, values in sorted(trial_ttfts.items())}

def get_trial_gaps(trial_ttfts):
    gaps = {}

    for trial, ttfts in trial_ttfts.items():
        gaps[trial] = [ttfts[i + 1] - ttfts[i] for i in range(len(ttfts) - 1)]

    return gaps

def print_trial_gap_comparison(baseline_batches, candidate_batches):
    threshold_ms = 300
    runs = {
        "Baseline": baseline_batches,
        "Candidate": candidate_batches,
    }
    ttfts = {label: get_trial_ttfts(batches) for label, batches in runs.items()}
    gaps = {label: get_trial_gaps(values) for label, values in ttfts.items()}
    trials = sorted({batch["trial"] for batches in runs.values() for batch in batches})

    print("\nLong workload, concurrency 8: TTFT gaps by trial (ms)")
    print("Threshold >300 ms is exploratory, not a proven scheduling boundary.")
    for label, batches in runs.items():
        budget = batches[0].get("max_num_batched_tokens") if batches else None
        print(
            f"{label} scheduling budget: {budget if budget is not None else 'unknown'}"
        )
    if not trials:
        print("No long-prompt trials found.")
        return

    comparable_trials = matching_counts = matching_positions = 0
    for trial in trials:
        print(f"\nTrial {trial}:")
        large_gap_positions = {}
        for label in runs:
            values = ttfts[label].get(trial, [])
            trial_gaps = gaps[label].get(trial, [])
            positions = [
                index
                for index, gap in enumerate(trial_gaps, start=1)
                if gap > threshold_ms
            ]
            large_gap_positions[label] = positions
            gap_display = ", ".join(f"{gap:.2f}" for gap in trial_gaps)
            apparent_groups = len(positions) + 1 if values else 0
            print(f"  {label} gaps (ms): [{gap_display}]")
            print(
                f"    Large gaps (>300 ms): {len(positions)}; "
                f"apparent groups: {apparent_groups}; valid TTFTs: {len(values)}/8"
            )

        if not all(len(ttfts[label].get(trial, [])) == 8 for label in runs):
            print(
                "  Pattern comparison: N/A (requires eight valid TTFTs in both runs)."
            )
            continue
        baseline_positions = large_gap_positions["Baseline"]
        candidate_positions = large_gap_positions["Candidate"]
        same_positions = baseline_positions == candidate_positions
        comparable_trials += 1
        matching_counts += len(baseline_positions) == len(candidate_positions)
        matching_positions += same_positions
        print(
            f"  Pattern comparison: {'same' if same_positions else 'different'} "
            f">300 ms gap positions; baseline={baseline_positions}, "
            f"candidate={candidate_positions}"
        )

    print(
        f"\nAcross {comparable_trials} complete trial pairs: "
        f"{matching_counts} have the same large-gap count; "
        f"{matching_positions} have the same large-gap positions."
    )

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
    groups = sorted(key for key in baseline.keys() | candidate.keys() if key[1] == 8)
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
            baseline_display = (
                f"{baseline_value:.2f}" if baseline_value is not None else "N/A"
            )
            candidate_display = (
                f"{candidate_value:.2f}" if candidate_value is not None else "N/A"
            )
            change_display = f"{change:+.2f}%" if change is not None else "N/A"
            print(
                f"{metric:<28} {baseline_display:>14} "
                f"{candidate_display:>14} {change_display:>12}"
            )

    print(
        "\nPositive latency changes are worse; positive throughput changes are better."
    )

    print_trial_gap_comparison(
        baseline.get(("long", 8), []), candidate.get(("long", 8), [])
    )

if __name__ == "__main__":
    main()
