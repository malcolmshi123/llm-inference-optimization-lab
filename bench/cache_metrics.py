import argparse
import json
import math
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families

from server_config import MODEL_NAME

COUNTERS = {
    "prefix_queries_tokens": "vllm:prefix_cache_queries_total",
    "prefix_hits_tokens": "vllm:prefix_cache_hits_total",
    "new_prefill_tokens": "vllm:request_prefill_kv_computed_tokens_sum",
    "preemptions": "vllm:num_preemptions_total"
}

def parse_metrics(text: str) -> dict:
    samples = {}

    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            key = sample.name, tuple(sorted(sample.labels.items()))

            if key in samples:
                raise ValueError(f"Duplicate metric series: {sample.name}")

            samples[key] = sample.value

    return samples

def series(samples: dict, name: str) -> dict:
    values = {labels: value for (metric, labels), value in samples.items() if metric == name}

    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError(f"Invalid metric value: {name}")

    return values

def process_identity(samples: dict) -> float:
    values = series(samples, "process_start_time_seconds")

    if len(values) != 1:
        raise ValueError("A single server process identity is required")

    return next(iter(values.values()))

def cache_configuration(samples: dict, expected: bool) -> dict:
    configs = series(samples, "vllm:cache_config_info")

    if len(configs) != 1:
        raise ValueError("A single cache_config_info series is required to verify the treatment")

    labels, value = next(iter(configs.items()))
    config = dict(labels)

    if value != 1 or config.get("enable_prefix_caching") != str(expected):
        raise ValueError("Running prefix caching does not match the expected treatment")

    if float(config.get("gpu_memory_utilization", "nan")) != 0.8:
        raise ValueError("Running GPU memory target does not match 0.80")

    return config

def is_idle(samples: dict) -> bool:
    running = series(samples, "vllm:num_requests_running")
    waiting = series(samples, "vllm:num_requests_waiting")

    if not running or running.keys() != waiting.keys():
        raise ValueError("Matching running/waiting request gauges are required")

    if any(dict(labels).get("model_name") != MODEL_NAME for labels in running):
        raise ValueError("Request gauges do not match the study model")

    return not any(running.values()) and not any(waiting.values())

def interval_summary(before: dict, after: dict) -> dict:
    if process_identity(before) != process_identity(after):
        raise ValueError("Server restarted during the metric interval")

    if series(before, "vllm:cache_config_info") != series(after, "vllm:cache_config_info"):
        raise ValueError("Cache configuration changed during the metric interval")

    result = {}

    for field, name in COUNTERS.items():
        initial, final = series(before, name), series(after, name)

        if not initial or not final:
            result[field] = None
            continue

        if initial.keys() != final.keys():
            raise ValueError(f"Incompatible metric labels: {name}")

        if any(dict(labels).get("model_name") != MODEL_NAME for labels in initial):
            raise ValueError(f"Counter does not match the study model: {name}")

        created = name.removesuffix("_total").removesuffix("_sum") + "_created"

        if series(before, created) != series(after, created):
            raise ValueError(f"Counter creation time changed: {name}")

        changes = [final[labels] - value for labels, value in initial.items()]

        if any(value < 0 for value in changes):
            raise ValueError(f"Counter decreased: {name}")

        result[field] = sum(changes)

    queries, hits = result["prefix_queries_tokens"], result["prefix_hits_tokens"]

    if queries is not None and hits is not None:
        if series(before, COUNTERS["prefix_queries_tokens"]).keys() != series(before, COUNTERS["prefix_hits_tokens"]).keys():
            raise ValueError("Prefix query/hit labels do not match")

        if hits > queries:
            raise ValueError("Prefix hits exceed queries")

    result["token_cache_hit_rate"] = hits / queries if queries and hits is not None else None

    return result

def main():
    parser = argparse.ArgumentParser(description="Inspect prefix-cache counter changes between two saved snapshots.")
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()

    try:
        summary = interval_summary(
            parse_metrics(args.before.read_text(encoding="utf-8")),
            parse_metrics(args.after.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(summary, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()
