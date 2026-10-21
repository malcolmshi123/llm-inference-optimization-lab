"""Inspect server timing observations in saved vLLM metric windows.

Means are computed from histogram sum/count deltas, weighted by observations
across engines. TPOT is the mean of per-request TPOT observations, rather than
a pooled per-token average. These values are neither client latency percentiles
nor GPU kernel timings; the server phases need not sum to client E2E latency.
"""

import argparse
import json
import math
from pathlib import Path

from bench.cache_metrics import counter_changes, parse_metrics, process_identity, series

TIMING_METRICS = {
    "queue": "vllm:request_queue_time_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "decode": "vllm:request_decode_time_seconds",
    "time_per_output_token": "vllm:request_time_per_output_token_seconds",
}


def _histogram_summary(before: dict, after: dict, metric: str) -> dict:
    sums = counter_changes(before, after, metric + "_sum")
    counts = counter_changes(before, after, metric + "_count")

    if sums is None or counts is None:
        return {
            "observations": None,
            "total_seconds": None,
            "mean_ms": None,
        }

    if sums.keys() != counts.keys():
        raise ValueError(f"Histogram sum/count labels do not match: {metric}")

    if any(
        value % 1
        for snapshot in (before, after)
        for value in series(snapshot, metric + "_count").values()
    ):
        raise ValueError(f"Histogram observation counts must be integers: {metric}")

    if any(counts[labels] == 0 and value != 0 for labels, value in sums.items()):
        raise ValueError(f"Histogram sum increased without observations: {metric}")

    total_seconds = sum(sums.values())
    observations = sum(counts.values())
    mean_ms = total_seconds / observations * 1000 if observations else None

    if (
        not math.isfinite(total_seconds)
        or not math.isfinite(observations)
        or (mean_ms is not None and not math.isfinite(mean_ms))
    ):
        raise ValueError(f"Non-finite timing aggregate: {metric}")

    return {
        "observations": int(observations),
        "total_seconds": total_seconds,
        "mean_ms": mean_ms,
    }


def timing_summary(before: dict, after: dict) -> dict:
    """Summarize one continuous server window; unavailable metrics stay None."""
    if process_identity(before) != process_identity(after):
        raise ValueError("Server restarted during the metric interval")

    if series(before, "vllm:cache_config_info") != series(
        after, "vllm:cache_config_info"
    ):
        raise ValueError("Cache configuration changed during the metric interval")

    return {
        field: _histogram_summary(before, after, metric)
        for field, metric in TIMING_METRICS.items()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inspect server timing histogram changes between saved snapshots.",
        epilog="Means describe server observations, not client percentiles or GPU kernel timings.",
    )
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()

    try:
        summary = timing_summary(
            parse_metrics(args.before.read_text(encoding="utf-8")),
            parse_metrics(args.after.read_text(encoding="utf-8")),
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
