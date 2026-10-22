import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from bench.cache_metrics import (
    cache_configuration,
    interval_summary,
    is_idle,
    parse_metrics,
    process_identity,
    validate_completed_requests,
)
from bench.cache_study import startup_command, write_json
from bench.cache_workloads import SCENARIOS, load_manifest
from bench.plot_style import BLUE, FIGURE_STYLE, ORANGE, label_figure
from bench.server_timings import timing_summary

LATENCIES = ("ttft_p50_ms", "ttft_p95_ms", "e2e_p50_ms", "e2e_p95_ms")
PERFORMANCE = (*LATENCIES, "requests_per_s", "output_tokens_per_s")

def number(value, field: str, *, positive=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(
            f"Invalid {field}: expected a finite {'positive' if positive else 'nonnegative'} number"
        )

    return value

def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)

    if parsed.tzinfo is None:
        raise ValueError("Timestamps must include a timezone")

    return parsed

def validate_request(request: dict, *, prompt_tokens=None, limit=128):
    if type(request.get("success")) is not bool:
        raise ValueError("Request success must be a boolean")

    if not request["success"]:
        return

    for field in ("ttft_ms", "e2e_ms"):
        number(request.get(field), field)

    if request["e2e_ms"] < request["ttft_ms"]:
        raise ValueError("E2E latency cannot be less than TTFT")

    if type(request.get("prompt_tokens")) is not int or request["prompt_tokens"] <= 0:
        raise ValueError("Invalid actual prompt token count")

    if prompt_tokens is not None and request["prompt_tokens"] != prompt_tokens:
        raise ValueError("Actual prompt token count disagrees with the workload")

    if (
        type(request.get("completion_tokens")) is not int
        or not 0 < request["completion_tokens"] <= limit
    ):
        raise ValueError("Invalid actual completion token count")

    if request.get("finish_reason") not in ("stop", "length"):
        raise ValueError("Successful request has no valid finish reason")

def summarize_scenario(scenario: dict, metrics: dict) -> dict:
    requests = [
        request for batch in scenario["batches"] for request in batch["results"]
    ]
    successful = [request for request in requests if request["success"]]
    elapsed = number(scenario["elapsed_s"], "measurement elapsed_s", positive=True)
    summary = {
        "scenario": scenario["name"],
        "concurrency": scenario["concurrency"],
        "attempted": len(requests),
        "successful": len(successful),
        "failed": len(requests) - len(successful),
        "elapsed_s": elapsed,
        "prompt_tokens": sum(request["prompt_tokens"] for request in successful),
        "completion_tokens": sum(
            request["completion_tokens"] for request in successful
        ),
        "finish_reasons": dict(
            Counter(request["finish_reason"] for request in successful)
        ),
        **metrics,
    }

    for field in ("ttft", "e2e"):
        values = [request[field + "_ms"] for request in successful]

        for percentile in (50, 95):
            summary[f"{field}_p{percentile}_ms"] = (
                float(np.percentile(values, percentile)) if values else None
            )

    summary["requests_per_s"] = len(successful) / elapsed
    summary["output_tokens_per_s"] = summary["completion_tokens"] / elapsed

    return summary

def load_run(directory: Path, expected: bool) -> tuple:
    manifest = load_manifest(directory / "workload.json")
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))

    if (
        not isinstance(run, dict)
        or run.get("schema_version") != 1
        or run.get("kind") != "cache_study"
        or run.get("status") != "completed"
    ):
        raise ValueError(
            f"{directory}: comparison requires a completed version 1 cache study"
        )

    if run.get("manifest_sha256") != manifest["sha256"]:
        raise ValueError("Run manifest digest disagrees with the saved workload")

    server = run["server"]
    identity = number(
        server["process_start_time_seconds"], "server process identity", positive=True
    )
    command = startup_command(
        (directory / "server.log").read_text(encoding="utf-8"), expected
    )

    if (
        server.get("expected_prefix_caching") is not expected
        or server.get("startup_command_from_log") != command
    ):
        raise ValueError(
            "Treatment or saved startup command does not match the server log"
        )

    for field in ("gpu_declared", "vllm_version_declared"):
        if not isinstance(server.get(field), str) or not server[field].strip():
            raise ValueError("Declared GPU and vLLM version are required")

    if not isinstance(run["client"].get("git_revision"), str) or not run["client"].get(
        "packages"
    ):
        raise ValueError("Client revision and package versions are required")

    started, ended = timestamp(run["started_at"]), timestamp(run["ended_at"])

    if ended <= started:
        raise ValueError("Invalid session timestamps")

    expected_names = {"session-start", "warmup-after"} | {
        f"{case}-c{concurrency}-{phase}"
        for case, concurrency in SCENARIOS
        for phase in ("before", "after")
    }

    if set(run["snapshots"]) != expected_names:
        raise ValueError("Missing or unexpected metric snapshots")

    samples = {}

    for name, reference in run["snapshots"].items():
        if reference["path"] != name + ".prom":
            raise ValueError("Snapshot path does not match its phase")

        observed_at = timestamp(reference["captured_at"])
        observed = parse_metrics(
            (directory / reference["path"]).read_text(encoding="utf-8")
        )

        if (
            not started <= observed_at <= ended
            or process_identity(observed) != identity
            or not is_idle(observed)
        ):
            raise ValueError(
                "Snapshot does not belong to an idle, continuous server session"
            )

        config = cache_configuration(observed, expected)

        if config != server.get("observed_cache_config"):
            raise ValueError("Saved cache configuration disagrees with raw snapshots")

        samples[name] = observed

    warmup = run["warmup"]

    if (
        warmup.get("max_tokens") != 128
        or len(warmup.get("messages", [])) != 8
        or len(warmup.get("results", [])) != 8
    ):
        raise ValueError("Incomplete engine warmup")

    for request in warmup["results"]:
        validate_request(request)

        if not request["success"]:
            raise ValueError("Engine warmup contains failed requests")

    if (
        not started
        <= timestamp(warmup["started_at"])
        <= timestamp(warmup["ended_at"])
        <= ended
    ):
        raise ValueError("Invalid warmup phase timestamps")

    if [(s["case"], s["concurrency"]) for s in run["scenarios"]] != list(SCENARIOS):
        raise ValueError("Run is missing one or more ordered scenarios")

    summaries = []
    previous_end = timestamp(run["snapshots"]["warmup-after"]["captured_at"])

    if previous_end < timestamp(warmup["ended_at"]):
        raise ValueError("Warmup snapshot precedes completed warmup")

    for source, scenario in zip(manifest["scenarios"], run["scenarios"]):
        name = source["name"]
        preparation_start = previous_end

        if scenario.get("status") != "completed" or scenario.get("name") != name:
            raise ValueError("Incomplete or mismatched scenario")

        before, after = name + "-before", name + "-after"

        if (
            scenario.get("before_snapshot") != before
            or scenario.get("after_snapshot") != after
        ):
            raise ValueError("Scenario snapshots do not match the measured phase")

        before_time = timestamp(run["snapshots"][before]["captured_at"])
        after_time = timestamp(run["snapshots"][after]["captured_at"])

        if (
            not previous_end
            <= before_time
            <= timestamp(scenario["started_at"])
            <= timestamp(scenario["ended_at"])
            <= after_time
        ):
            raise ValueError(
                "Measured scenario phases overlap preparation or another scenario"
            )

        previous_end = after_time
        primer = scenario.get("priming")

        if source["primer"]:
            if not primer or len(primer.get("results", [])) != 1:
                raise ValueError("Shared-prefix scenario is missing its primer")

            request = primer["results"][0]
            validate_request(
                request, prompt_tokens=source["primer"]["prompt_tokens"], limit=1
            )

            if (
                not request["success"]
                or not preparation_start
                <= timestamp(primer["started_at"])
                <= timestamp(primer["ended_at"])
                <= before_time
            ):
                raise ValueError("Priming must succeed before the measurement baseline")
        elif primer is not None:
            raise ValueError("Unique-prefix scenario unexpectedly contains a primer")

        expected_batches = [
            source["requests"][offset : offset + source["concurrency"]]
            for offset in range(0, len(source["requests"]), source["concurrency"])
        ]

        if len(scenario["batches"]) != len(expected_batches):
            raise ValueError("Incomplete measured request batches")

        for selected, batch in zip(expected_batches, scenario["batches"]):
            number(batch["batch_time_s"], "batch time", positive=True)

            if batch["request_ids"] != [r["request_id"] for r in selected] or len(
                batch["results"]
            ) != len(selected):
                raise ValueError("Measured requests do not match the saved workload")

            for request, result in zip(selected, batch["results"]):
                validate_request(result, prompt_tokens=request["prompt_tokens"])

                if (
                    result["success"]
                    and result["e2e_ms"] > batch["batch_time_s"] * 1000 + 0.001
                ):
                    raise ValueError("Request latency exceeds its batch duration")

        elapsed = number(scenario["elapsed_s"], "measurement elapsed_s", positive=True)

        if elapsed + 0.000001 < sum(
            batch["batch_time_s"] for batch in scenario["batches"]
        ):
            raise ValueError("Measurement elapsed time excludes measured batch time")

        metrics = interval_summary(samples[before], samples[after])

        if scenario.get("metrics") != metrics:
            raise ValueError("Saved metric deltas disagree with raw snapshots")

        validate_completed_requests(
            samples[before],
            samples[after],
            [request for batch in scenario["batches"] for request in batch["results"]],
        )
        summary = summarize_scenario(scenario, metrics)

        for phase, timing in timing_summary(samples[before], samples[after]).items():
            summary[f"server_{phase}_mean_ms"] = timing["mean_ms"]
            summary[f"server_{phase}_observations"] = timing["observations"]

        summaries.append(summary)

    return run, manifest, summaries

def compare_runs(
    round_1_off: Path,
    round_1_on: Path,
    round_2_off: Path,
    round_2_on: Path,
    *,
    require_cache_metrics: bool = False,
) -> dict:
    directories = [round_1_off, round_1_on, round_2_on, round_2_off]
    loaded = [
        load_run(directory, expected)
        for directory, expected in zip(directories, (False, True, True, False))
    ]
    identities = [run["server"]["process_start_time_seconds"] for run, _, _ in loaded]

    if len(set(identities)) != 4:
        raise ValueError(
            "All four treatment sessions require different server processes"
        )

    reference = loaded[0][0]["client"]
    controls = (
        "model",
        "revision",
        "chat_template_sha256",
        "fixture_sha256",
        "transformers_version",
        "tokenizer_class",
        "generation",
        "count_per_scenario",
    )

    for index, (run, manifest, _) in enumerate(loaded):
        if any(
            run["server"][field] != loaded[0][0]["server"][field]
            for field in ("gpu_declared", "vllm_version_declared")
        ):
            raise ValueError("Declared GPU or vLLM version changed between sessions")

        if any(
            run["client"].get(key) != reference.get(key)
            for key in ("git_revision", "python", "packages")
        ):
            raise ValueError(
                "Client code revision or software versions changed between sessions"
            )

        if any(manifest[key] != loaded[0][1][key] for key in controls):
            raise ValueError("Workload or tokenizer controls changed between rounds")

        if index and timestamp(loaded[index - 1][0]["ended_at"]) > timestamp(
            run["started_at"]
        ):
            raise ValueError("Sessions do not follow the required off/on, on/off order")

        previous = run["server"].get("previous_process_start_time_seconds")

        if index and previous is not None and previous != identities[index - 1]:
            raise ValueError(
                "Previous-session identity does not match the preceding treatment"
            )

    if loaded[0][1]["seed"] == loaded[2][1]["seed"]:
        raise ValueError("Use different workload seeds between rounds")

    rows, changes = [], []

    for round_number, (off_index, on_index) in enumerate(((0, 1), (3, 2)), start=1):
        off, on = loaded[off_index], loaded[on_index]

        if (
            off[1]["sha256"] != on[1]["sha256"]
            or off[0]["warmup"]["messages"] != on[0]["warmup"]["messages"]
        ):
            raise ValueError(
                "Paired treatments must replay identical workloads and warmup payloads"
            )

        for cache, item in (("off", off), ("on", on)):
            rows.extend(
                {
                    "round": round_number,
                    "order": "off-on" if round_number == 1 else "on-off",
                    "seed": item[1]["seed"],
                    "cache": cache,
                    **row,
                }
                for row in item[2]
            )

        for baseline, candidate in zip(off[2], on[2]):
            differences = {
                key: (candidate[key] - baseline[key]) / baseline[key] * 100
                if baseline[key] not in (None, 0) and candidate[key] is not None
                else None
                for key in PERFORMANCE
            }
            changes.append(
                {
                    "round": round_number,
                    "scenario": baseline["scenario"],
                    "cache_on_change_pct": differences,
                }
            )

    if require_cache_metrics:
        required = (
            "prefix_queries_tokens",
            "prefix_hits_tokens",
            "new_prefill_tokens",
        )
        for row in rows:
            missing = [field for field in required if row[field] is None]
            if missing:
                raise ValueError(
                    f"Round {row['round']} cache-{row['cache']} "
                    f"{row['scenario']}: missing required cache metrics: "
                    + ", ".join(missing)
                )

    notes = [
        "Two paired rounds are exploratory session replications; "
        "requests within a session are not independent replications.",
        "Percentiles describe successful requests only; token totals "
        "and output throughput also count successful requests only.",
        "Throughput uses the full measured elapsed time, including "
        "failed-request time and client checkpoint overhead.",
        "Server timing means use histogram sum/count deltas with observation "
        "counts reported separately. TPOT averages per-request observations. "
        "These are not client percentiles or GPU kernel timings, and the "
        "server phases need not sum to client E2E latency.",
        "Warmup and priming are excluded. Missing telemetry is unavailable. "
        "No bootstrap confidence intervals are computed.",
        "GPU identity and server software versions are declared settings "
        "supported by startup logs, not independently measured runtime identities.",
    ]

    if require_cache_metrics:
        notes.append(
            "Required cache query, hit, and new-prefill counters are "
            "present in every measured interval."
        )

    if any(run["client"].get("git_dirty") for run, _, _ in loaded):
        notes.append(
            "At least one client worktree was dirty; matching Git revisions "
            "do not prove identical uncommitted code."
        )

    if loaded[0][1]["count_per_scenario"] != 32:
        notes.append(
            "Pilot-sized workload: this comparison does not use "
            "the planned 32 requests per scenario."
        )

    return {
        "schema_version": 1,
        "kind": "cache_comparison",
        "inputs": [str(path.resolve()) for path in directories],
        "notes": notes,
        "rows": rows,
        "changes": changes,
    }

def write_figures(comparison: dict, directory: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(FIGURE_STYLE):
        _write_figures(comparison, directory)

def _write_figures(comparison: dict, directory: Path):
    import matplotlib.pyplot as plt

    names = [f"{case}-c{concurrency}" for case, concurrency in SCENARIOS]
    labels = [f"{case.title()}\nc={concurrency}" for case, concurrency in SCENARIOS]
    x = np.arange(len(names))
    colors = {
        "off": BLUE,
        "on": ORANGE,
    }
    rows = {
        (row["round"], row["cache"], row["scenario"]): row for row in comparison["rows"]
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 7))

    for round_number in (1, 2):
        for column, metric in enumerate(("ttft", "e2e")):
            axis = axes[round_number - 1, column]

            for cache in ("off", "on"):
                for percentile in (50, 95):
                    values = [
                        rows[round_number, cache, name][f"{metric}_p{percentile}_ms"]
                        for name in names
                    ]
                    axis.plot(
                        x,
                        [value if value is not None else np.nan for value in values],
                        color=colors[cache],
                        linestyle="-" if percentile == 50 else "--",
                        marker="o" if percentile == 50 else "D",
                        markerfacecolor=colors[cache],
                        markeredgecolor=colors[cache],
                        label=f"Cache {cache} p{percentile}",
                    )

            axis.set(
                title=(
                    f"Round {round_number} · "
                    f"{'off → on' if round_number == 1 else 'on → off'} · "
                    f"{'TTFT' if metric == 'ttft' else 'End-to-end'}"
                ),
                ylabel="Latency (ms)",
                xticks=x,
                xticklabels=labels,
            )
            axis.set_ylim(bottom=0)
            axis.grid(axis="y")

    figure.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(0.075, 0.87),
        ncol=4,
    )
    sizes = sorted({row["successful"] for row in comparison["rows"]})
    sample_size = str(sizes[0]) if len(sizes) == 1 else f"{sizes[0]}–{sizes[-1]}"
    label_figure(
        figure,
        "Prefix-cache latency by scenario",
        "Approximately 2048 input tokens · Output limit 128 · Two paired rounds",
        f"n = {sample_size} successful requests per cell · "
        "p95 is descriptive · No confidence intervals",
    )
    figure.tight_layout(rect=(0.01, 0.06, 0.99, 0.84))

    for extension in ("png", "svg"):
        figure.savefig(directory / f"01_cache_latency.{extension}", dpi=160)

    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(12, 7))

    for round_number in (1, 2):
        for column, (field, scale, title) in enumerate(
            (
                ("token_cache_hit_rate", 100, "Token cache hit rate (%)"),
                ("new_prefill_tokens", 0.001, "New prefill tokens (thousands)"),
            )
        ):
            axis = axes[round_number - 1, column]

            for cache, offset in (("off", -0.18), ("on", 0.18)):
                values = [rows[round_number, cache, name][field] for name in names]
                axis.bar(
                    x + offset,
                    [
                        value * scale if value is not None else np.nan
                        for value in values
                    ],
                    width=0.36,
                    color=colors[cache],
                    label=f"Cache {cache}",
                )

                for position, value in zip(x + offset, values):
                    if value is None:
                        axis.text(
                            position,
                            0,
                            "N/A",
                            ha="center",
                            va="bottom",
                            rotation=90,
                            fontsize=8,
                        )

            axis.set(
                title=f"Round {round_number} · {title}", xticks=x, xticklabels=labels
            )
            axis.set_ylim(bottom=0)

            if field == "token_cache_hit_rate":
                axis.set_ylim(0, 105)

            axis.grid(axis="y")

    figure.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(0.075, 0.87),
        ncol=2,
    )
    label_figure(
        figure,
        "Cache reuse and prefill work",
        "Measured-window counter changes · Warmup and priming excluded",
        "N/A = unavailable telemetry or no prefix queries",
    )
    figure.tight_layout(rect=(0.01, 0.06, 0.99, 0.84))

    for extension in ("png", "svg"):
        figure.savefig(directory / f"02_cache_prefill.{extension}", dpi=160)

    plt.close(figure)
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), sharey="col")
    phases = (
        ("queue", "Queue", "Mean per request (ms)"),
        ("prefill", "Prefill", "Mean per request (ms)"),
        ("decode", "Decode", "Mean per request (ms)"),
        ("time_per_output_token", "TPOT", "Mean TPOT (ms/token)"),
    )

    for round_number in (1, 2):
        for column, (phase, title, unit) in enumerate(phases):
            axis = axes[round_number - 1, column]

            for cache, offset in (("off", -0.18), ("on", 0.18)):
                values = [
                    rows[round_number, cache, name].get(f"server_{phase}_mean_ms")
                    for name in names
                ]
                axis.bar(
                    x + offset,
                    [value if value is not None else np.nan for value in values],
                    width=0.36,
                    color=colors[cache],
                    label=f"Cache {cache}",
                )

                for position, value in zip(x + offset, values):
                    if value is None:
                        axis.text(
                            position,
                            0.02,
                            "N/A",
                            transform=axis.get_xaxis_transform(),
                            ha="center",
                            va="bottom",
                            rotation=90,
                            fontsize=8,
                        )

            axis.set(
                title=f"Round {round_number} · {title}",
                ylabel=unit,
                xticks=x,
                xticklabels=labels,
            )
            axis.grid(axis="y")

    for column, (phase, _, _) in enumerate(phases):
        values = [row.get(f"server_{phase}_mean_ms") for row in comparison["rows"]]
        maximum = max((value for value in values if value is not None), default=0)
        axes[0, column].set_ylim(0, maximum * 1.05 if maximum else 1)

    counts = sorted(
        {
            count
            for row in comparison["rows"]
            for phase, _, _ in phases
            if (count := row.get(f"server_{phase}_observations")) is not None
        }
    )
    observation_count = "unavailable"

    if counts:
        observation_count = (
            str(counts[0]) if len(counts) == 1 else f"{counts[0]}–{counts[-1]}"
        )

    figure.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(0.075, 0.87),
        ncol=2,
    )
    label_figure(
        figure,
        "Server-side timing by round",
        "Histogram means · TPOT averages per-request observations",
        f"n = {observation_count} per histogram · "
        "N/A = unavailable · Server means, not GPU timings",
    )
    figure.tight_layout(rect=(0.01, 0.06, 0.99, 0.84))

    for extension in ("png", "svg"):
        figure.savefig(directory / f"03_server_timings.{extension}", dpi=160)

    plt.close(figure)

def export_comparison(comparison: dict, directory: Path):
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "comparison.json", comparison)

    with (directory / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison["rows"][0]))
        writer.writeheader()
        writer.writerows(
            {
                **row,
                "finish_reasons": json.dumps(row["finish_reasons"], sort_keys=True),
            }
            for row in comparison["rows"]
        )

    write_figures(comparison, directory)

def main():
    parser = argparse.ArgumentParser(
        description="Compare two manually executed cache-study pairs and draw three figures."
    )

    for name in (
        "round-1-off",
        "round-1-on",
        "round-2-off",
        "round-2-on",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)

    parser.add_argument(
        "--require-cache-metrics",
        action="store_true",
        help="Require cache query, hit, and new-prefill counters in every measured interval.",
    )

    args = parser.parse_args()

    try:
        comparison = compare_runs(
            args.round_1_off,
            args.round_1_on,
            args.round_2_off,
            args.round_2_on,
            require_cache_metrics=args.require_cache_metrics,
        )
        export_comparison(comparison, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    for row in comparison["rows"]:
        ttft = f"{row['ttft_p50_ms']:.2f}" if row["ttft_p50_ms"] is not None else "N/A"
        print(
            f"Round {row['round']} | {row['scenario']:<10} | "
            f"cache {row['cache']:<3} | TTFT p50 {ttft} ms | "
            f"{row['requests_per_s']:.3f} req/s | "
            f"{row['output_tokens_per_s']:.2f} output tok/s | "
            f"failures {row['failed']}/{row['attempted']}"
        )

    for note in comparison["notes"]:
        print(note)

    print(f"Saved comparison data and figures to {args.output_dir}")

if __name__ == "__main__":
    main()
