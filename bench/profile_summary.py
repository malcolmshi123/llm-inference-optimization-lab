"""Summarize one saved single-GPU profiling pair without contacting a server."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from bench.cache_metrics import (
    cache_configuration,
    interval_summary,
    is_idle,
    parse_metrics,
    process_identity,
    validate_completed_requests,
)
from bench.cache_study import write_json
from bench.cache_workloads import load_manifest
from bench.compare_cache import validate_request
from server_config import ServerConfig

ANNOTATION = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
TIMELINE_GAP_MS = 0.5


def union_intervals(
    intervals: list[tuple[float, float]], *, gap: float = 0
) -> list[tuple[float, float]]:
    """Merge intervals; a positive gap is reserved for display coalescing."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + gap:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def event_number(event: dict, key: str) -> float:
    value = event.get(key)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Invalid trace {key}: expected a finite number")
    if key == "dur" and value < 0:
        raise ValueError("Negative trace duration")
    return value


def summarize_trace(trace: dict) -> dict:
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("Expected a Chrome traceEvents array")
    kernels = defaultdict(list)
    intervals = []
    annotations = []
    devices = set()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Trace events must be objects")
        name = event.get("name", "")
        match = ANNOTATION.fullmatch(name) if isinstance(name, str) else None
        is_annotation = event.get("cat") == "user_annotation" and match is not None
        if event.get("cat") != "kernel" and not is_annotation:
            continue
        if event.get("ph") != "X":
            raise ValueError(
                "Only complete (X) kernel and execution events are supported"
            )
        start, duration = event_number(event, "ts"), event_number(event, "dur")
        if not math.isfinite(start + duration):
            raise ValueError("Non-finite trace interval endpoint")
        if event.get("cat") == "kernel":
            if duration == 0:
                continue
            name = event.get("name")
            device = event.get("args", {}).get("device")
            if not isinstance(name, str) or not name or type(device) is not int:
                raise ValueError("GPU kernels require a name and integer device ID")
            devices.add(device)
            kernels[name].append(duration)
            intervals.append((start, start + duration))
        else:
            context_requests, context_tokens, generation_requests, generation_tokens = (
                int(value) for value in match.groups()
            )
            phase = "empty"
            if context_requests:
                phase = "mixed" if generation_requests else "context_only"
            elif generation_requests:
                phase = "generation_only"
            annotations.append(
                {
                    "name": event["name"],
                    "phase": phase,
                    "start_us": start,
                    "duration_ms": duration / 1000,
                    "context_requests": context_requests,
                    "context_tokens": context_tokens,
                    "generation_requests": generation_requests,
                    "generation_tokens": generation_tokens,
                }
            )
    if len(devices) > 1:
        raise ValueError("This analysis supports one GPU device per worker trace")
    rows = [
        {
            "name": name,
            "calls": len(durations),
            "total_ms": math.fsum(durations) / 1000,
            "mean_ms": math.fsum(durations) / len(durations) / 1000,
        }
        for name, durations in kernels.items()
    ]
    rows.sort(key=lambda row: (-row["total_ms"], row["name"]))
    merged = union_intervals(intervals)
    origin = min(
        [start for start, _ in intervals] + [row["start_us"] for row in annotations],
        default=0,
    )
    for row in annotations:
        row["start_ms"] = (row.pop("start_us") - origin) / 1000
    annotations.sort(key=lambda row: (row["start_ms"], row["name"]))
    return {
        "kernel_count": sum(row["calls"] for row in rows),
        "summed_kernel_ms": math.fsum(row["total_ms"] for row in rows),
        "kernel_span_ms": (merged[-1][1] - merged[0][0]) / 1000 if merged else None,
        "kernel_union_ms": math.fsum(end - start for start, end in merged) / 1000,
        "timeline_gap_ms": TIMELINE_GAP_MS,
        "timeline_kernel_intervals_ms": [
            ((start - origin) / 1000, (end - start) / 1000)
            for start, end in union_intervals(merged, gap=TIMELINE_GAP_MS * 1000)
        ],
        "device_properties": trace.get("deviceProperties", []),
        "kernels": rows,
        "annotations": annotations,
    }


def load_capture(directory: Path, expected: bool) -> dict:
    run = json.loads((directory / "capture.json").read_text(encoding="utf-8"))
    if (
        run.get("schema_version") != 1
        or run.get("kind") != "profile_capture"
        or run.get("status") != "completed"
        or run.get("scenario") != "shared-c8"
        or run["server"]["expected_prefix_caching"] is not expected
        or not run["profiler"]["start_completed"]
        or not run["profiler"]["stop_completed"]
    ):
        raise ValueError(
            "Expected a completed shared-c8 profiling capture with matching treatment"
        )
    config = ServerConfig(
        prefix_caching=expected, profile_session=run["profiler"]["session"]
    )
    if (
        config.profile_directory is None
        or run["server"]["startup_command_from_log"] != config.command()
    ):
        raise ValueError(
            "Capture startup configuration does not match its treatment/session"
        )
    manifest = load_manifest(directory / "workload.json")
    selected = manifest["scenarios"][-1]["requests"][:8]
    results = run["captured_batch"]["results"]
    if (
        manifest["sha256"] != run["manifest_sha256"]
        or len(selected) != 8
        or len(results) != 8
        or run["request_ids"] != [request["request_id"] for request in selected]
    ):
        raise ValueError("Capture requests do not match the saved workload")
    for request, result in zip(selected, results):
        validate_request(
            result, prompt_tokens=request["prompt_tokens"], limit=request["max_tokens"]
        )
    before, after = [
        parse_metrics((directory / f"capture-{phase}.prom").read_text(encoding="utf-8"))
        for phase in ("before", "after")
    ]
    for samples in (before, after):
        cache_configuration(samples, expected)
        if (
            not is_idle(samples)
            or process_identity(samples) != run["server"]["process_start_time_seconds"]
        ):
            raise ValueError(
                "Capture metric windows are busy or have inconsistent server identities"
            )
    validate_completed_requests(before, after, results)
    if interval_summary(before, after) != run["metrics"]:
        raise ValueError("Capture metrics disagree with raw snapshots")
    receipt = json.loads((directory / "trace-check.json").read_text(encoding="utf-8"))
    if receipt.get("kind") != "profile_trace_check":
        raise ValueError("Expected a trace download-check receipt")
    inventory, workers, seen = [], [], set()
    for entry in receipt["traces"]:
        path = (directory / entry["path"]).resolve()
        if not path.is_relative_to((directory / "traces").resolve()) or path in seen:
            raise ValueError(
                "Trace receipt paths must be distinct files within traces/"
            )
        seen.add(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Trace hash disagrees with its download receipt")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            trace = summarize_trace(json.load(handle))
        if trace["kernel_count"] != entry["gpu_kernel_events"]:
            raise ValueError("Trace kernel count disagrees with its download receipt")
        inventory.append(
            {
                **entry,
                "device_properties": trace["device_properties"],
            }
        )
        if trace["kernel_count"]:
            workers.append(trace)
    if seen != {path.resolve() for path in (directory / "traces").rglob("*.json.gz")}:
        raise ValueError(
            "Trace receipt does not cover every downloaded compressed trace"
        )
    if len(workers) != 1 or not workers[0]["annotations"]:
        raise ValueError(
            "Expected one GPU worker trace containing execution annotations"
        )
    worker = workers[0]
    context_tokens = sum(row["context_tokens"] for row in worker["annotations"])
    if (
        run["metrics"]["new_prefill_tokens"] is not None
        and context_tokens != run["metrics"]["new_prefill_tokens"]
    ):
        raise ValueError(
            "Execution context tokens disagree with measured new-prefill tokens"
        )
    return {
        "cache": "on" if expected else "off",
        "input": str(directory.resolve()),
        "manifest_sha256": manifest["sha256"],
        "request_ids": run["request_ids"],
        "server_identity": run["server"]["process_start_time_seconds"],
        "completion_tokens": sum(result["completion_tokens"] for result in results),
        "metrics": run["metrics"],
        "trace_inventory": inventory,
        "context_tokens": context_tokens,
        **worker,
    }


def summarize_pair(off: Path, on: Path) -> dict:
    captures = [load_capture(off, False), load_capture(on, True)]
    left, right = captures
    if (
        left["manifest_sha256"] != right["manifest_sha256"]
        or left["request_ids"] != right["request_ids"]
    ):
        raise ValueError(
            "Profiling pair must use the same workload and selected requests"
        )
    if left["server_identity"] == right["server_identity"]:
        raise ValueError("Profiling pair must use fresh server processes")
    if left["device_properties"] != right["device_properties"]:
        raise ValueError("Profiling pair has different observed GPU properties")
    return {
        "schema_version": 1,
        "kind": "profile_summary",
        "notes": [
            "One profiled batch per treatment; output lengths can differ. "
            "No benchmark speedup is estimated.",
            "Kernel totals sum event durations and may double-count overlap. "
            "Union duration merges overlapping kernel intervals on one GPU; "
            "neither is SM utilization.",
            "GPU gaps under 0.5 ms are merged for timeline display only; "
            "exact kernel sums, spans, and union durations are unchanged.",
            "Chrome JSON ts/dur are microseconds, regardless of displayTimeUnit. "
            "Timelines are aligned independently within each worker trace.",
            "Execution annotations are host CPU scopes. Context-only, mixed, "
            "and generation-only labels describe scheduled work, not GPU phase "
            "durations. Kernels are not attributed to phases by timestamp overlap.",
            "Context request counts can repeat across chunked prefill iterations; "
            "generation tokens are scheduled tokens, not necessarily emitted "
            "completion tokens.",
        ],
        "captures": captures,
    }


def write_timeline(summary: dict, directory: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {
        "context_only": "#D55E00",
        "mixed": "#CC79A7",
        "generation_only": "#009E73",
        "empty": "#999999",
    }
    figure, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    for axis, capture in zip(axes, summary["captures"]):
        axis.broken_barh(
            capture["timeline_kernel_intervals_ms"], (0.7, 0.5), facecolors="#0072B2"
        )
        for phase, color in colors.items():
            ranges = [
                (row["start_ms"], row["duration_ms"])
                for row in capture["annotations"]
                if row["phase"] == phase
            ]
            axis.broken_barh(ranges, (1.7, 0.5), facecolors=color)
        axis.set(
            title=(
                f"Cache {capture['cache']} | {capture['kernel_count']:,} GPU kernels | "
                f"{capture['completion_tokens']} output tokens"
            ),
            yticks=[0.95, 1.95],
            yticklabels=["GPU kernels", "Host scopes"],
            ylim=(0.4, 2.5),
        )
        axis.grid(axis="x", alpha=0.2)
    axes[-1].set_xlabel(
        "Time from earliest GPU kernel / host execution scope in each trace (ms)"
    )
    axes[-1].set_xlim(left=0)
    figure.suptitle(
        "Shared-prefix profiling | one eight-request batch per treatment", y=0.99
    )
    figure.legend(
        handles=[
            Patch(color=color, label=phase.replace("_", " "))
            for phase, color in colors.items()
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
    )
    figure.text(
        0.5,
        0.02,
        "Host scopes describe scheduled work; GPU execution may outlive them. "
        "GPU lane merges recorded kernel intervals.\n"
        "Not GPU utilization or a prefill/decode GPU-time decomposition. "
        "Profiling overhead is included.\n"
        "GPU gaps under 0.5 ms are merged for display only; exact duration "
        "totals are preserved in summary.json.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.12, 1, 0.86))
    for extension in ("png", "svg"):
        figure.savefig(directory / f"01_gpu_timeline.{extension}", dpi=160)
    plt.close(figure)


def export_summary(summary: dict, directory: Path):
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "summary.json", summary)
    for name in ("kernels", "annotations"):
        rows = [
            {
                "cache": capture["cache"],
                **row,
            }
            for capture in summary["captures"]
            for row in capture[name]
        ]
        with (directory / f"{name}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_timeline(summary, directory)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize a saved single-GPU profiling pair offline."
    )
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = summarize_pair(args.off, args.on)
        export_summary(summary, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    for capture in summary["captures"]:
        print(
            f"Cache {capture['cache']}: {capture['kernel_count']:,} kernels, "
            f"{capture['summed_kernel_ms']:.3f} summed kernel ms, "
            f"{capture['context_tokens']:,} context tokens"
        )
    print(f"Saved offline profiling summary to {args.output_dir}")


if __name__ == "__main__":
    main()
