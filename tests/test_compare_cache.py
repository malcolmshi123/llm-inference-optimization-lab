import copy
import asyncio
import csv
import json
import subprocess
import sys
import unittest
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx

from bench.cache_metrics import cache_configuration, interval_summary, parse_metrics
from bench.cache_study import run_study, write_json
from bench.cache_workloads import create_manifest, manifest_digest
from bench.compare_cache import (
    compare_runs,
    export_comparison,
    load_run,
    summarize_scenario,
)
from bench.server_timings import TIMING_METRICS
from server_config import GPU_TYPE, VLLM_VERSION, ServerConfig
from tests.test_cache_metrics import metric_text
from tests.test_cache_study import StudyServer
from tests.test_cache_workloads import WordTokenizer
from tests.test_server_timings import histogram_text


def result(prompt_tokens, index=0, scale=1):
    return {
        "success": True,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": index + 1,
        "finish_reason": "stop",
        "ttft_ms": (index + 1) * 10 * scale,
        "e2e_ms": (index + 10) * 10 * scale,
    }


def create_run(directory: Path, manifest: dict, *, expected: bool, session: int):
    directory.mkdir(parents=True)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=session)
    identity = 100 + session

    def when(seconds):
        return (base + timedelta(seconds=seconds)).isoformat()

    command = ServerConfig(prefix_caching=expected).command()
    run = {
        "schema_version": 1,
        "kind": "cache_study",
        "status": "completed",
        "manifest_sha256": manifest["sha256"],
        "started_at": when(0),
        "ended_at": when(50),
        "client": {
            "git_revision": "a" * 40,
            "git_dirty": False,
            "python": "3.11.9",
            "packages": {
                "httpx": "0.28.1",
            },
        },
        "server": {
            "expected_prefix_caching": expected,
            "startup_command_from_log": command,
            "vllm_version_declared": VLLM_VERSION,
            "gpu_declared": GPU_TYPE,
            "process_start_time_seconds": identity,
            "previous_process_start_time_seconds": 99 + session if session else None,
        },
        "warmup": {
            "messages": [
                [
                    {
                        "role": "user",
                        "content": f"Unrelated warmup {manifest['seed']} request {index}",
                    }
                ]
                for index in range(8)
            ],
            "max_tokens": 128,
            "results": [result(100) for _ in range(8)],
            "started_at": when(0.1),
            "ended_at": when(0.5),
        },
        "snapshots": {},
        "scenarios": [],
    }
    tokens, hits = 0, 0
    completed = Counter(
        {
            "stop": 0,
        }
    )

    def snapshot(name, seconds):
        text = metric_text(
            tokens=tokens,
            hits=hits,
            prefill=tokens - hits,
            identity=identity,
            expected=expected,
            finished=completed,
        )
        (directory / (name + ".prom")).write_text(text, encoding="utf-8")
        run["snapshots"][name] = {
            "path": name + ".prom",
            "captured_at": when(seconds),
        }
        run["server"]["observed_cache_config"] = cache_configuration(
            parse_metrics(text), expected
        )

        return parse_metrics(text)

    snapshot("session-start", 0.05)
    tokens += 800
    completed["stop"] += 8
    snapshot("warmup-after", 0.6)

    for index, source in enumerate(manifest["scenarios"], start=1):
        seconds = index * 10
        primer = None

        if source["primer"]:
            tokens += source["primer"]["prompt_tokens"]
            completed["stop"] += 1
            primer = {
                "results": [result(source["primer"]["prompt_tokens"])],
                "started_at": when(seconds + 0.01),
                "ended_at": when(seconds + 0.05),
            }

        before_name, after_name = source["name"] + "-before", source["name"] + "-after"
        before = snapshot(before_name, seconds + 0.1)
        requests = [
            result(request["prompt_tokens"], index, 0.5 if expected else 1)
            for index, request in enumerate(source["requests"])
        ]
        tokens += sum(request["prompt_tokens"] for request in source["requests"])
        hits += (
            sum(request["shared_prefix_tokens"] or 0 for request in source["requests"])
            if expected
            else 0
        )
        completed["stop"] += len(requests)
        after = snapshot(after_name, seconds + 2.3)
        batches = []

        for offset in range(0, len(requests), source["concurrency"]):
            selected = source["requests"][offset : offset + source["concurrency"]]
            batches.append(
                {
                    "request_ids": [request["request_id"] for request in selected],
                    "results": requests[offset : offset + source["concurrency"]],
                    "batch_time_s": 0.2,
                }
            )

        run["scenarios"].append(
            {
                "name": source["name"],
                "case": source["case"],
                "concurrency": source["concurrency"],
                "status": "completed",
                "priming": primer,
                "started_at": when(seconds + 0.2),
                "ended_at": when(seconds + 2.2),
                "elapsed_s": 1 if expected else 2,
                "before_snapshot": before_name,
                "after_snapshot": after_name,
                "batches": batches,
                "metrics": interval_summary(before, after),
            }
        )

    write_json(directory / "workload.json", manifest)
    write_json(directory / "run.json", run)
    (directory / "server.log").write_text(
        "Starting vLLM server with command: " + " ".join(command), encoding="utf-8"
    )


def create_pairs(root: Path) -> list[Path]:
    first = create_manifest(WordTokenizer(), seed=42, count=4)
    second = create_manifest(WordTokenizer(), seed=43, count=4)
    directories = [
        root / name
        for name in ("round-1-off", "round-1-on", "round-2-off", "round-2-on")
    ]

    for directory, manifest, expected, session in zip(
        directories,
        (first, first, second, second),
        (False, True, False, True),
        (0, 1, 3, 2),
    ):
        create_run(directory, manifest, expected=expected, session=session)

    return directories


class CompareCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directories = create_pairs(self.root)

    def edit_run(self, index, mutate):
        path = self.directories[index] / "run.json"
        run = json.loads(path.read_text(encoding="utf-8"))
        mutate(run)
        write_json(path, run)

    def test_rounds_actual_tokens_percentiles_and_changes_remain_separate(self):
        comparison = compare_runs(*self.directories)
        row = comparison["rows"][0]

        self.assertEqual(len(comparison["rows"]), 16)
        self.assertEqual(len(comparison["changes"]), 8)
        self.assertEqual(row["ttft_p50_ms"], 25)
        self.assertAlmostEqual(row["ttft_p95_ms"], 38.5)
        self.assertEqual(row["attempted"], 4)
        self.assertEqual(row["completion_tokens"], 10)
        self.assertEqual(row["output_tokens_per_s"], 5)
        self.assertEqual(row["requests_per_s"], 2)
        self.assertEqual(comparison["rows"][8]["order"], "on-off")
        self.assertEqual(
            comparison["changes"][0]["cache_on_change_pct"]["ttft_p50_ms"], -50
        )
        self.assertEqual(
            comparison["changes"][0]["cache_on_change_pct"]["output_tokens_per_s"], 100
        )
        self.assertEqual(row["new_prefill_tokens"], row["prompt_tokens"])
        self.assertGreater(comparison["rows"][5]["token_cache_hit_rate"], 0.85)

    def test_failures_stay_in_attempts_and_elapsed_time(self):
        scenario = {
            "name": "unique-c1",
            "concurrency": 1,
            "elapsed_s": 2,
            "batches": [
                {
                    "results": [
                        result(2048),
                        {
                            "success": False,
                        },
                    ],
                }
            ],
        }
        summary = summarize_scenario(scenario, {})

        self.assertEqual(summary["attempted"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["requests_per_s"], 0.5)
        self.assertEqual(summary["output_tokens_per_s"], 0.5)
        self.assertEqual(summary["ttft_p50_ms"], 10)

    def test_mismatched_paired_workload_is_rejected(self):
        directory = self.directories[1]
        path = directory / "workload.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["scenarios"][0]["requests"][0]["messages"][0]["content"] += " changed"
        manifest["sha256"] = manifest_digest(manifest)
        write_json(path, manifest)
        self.edit_run(1, lambda run: run.update(manifest_sha256=manifest["sha256"]))

        with self.assertRaisesRegex(ValueError, "identical workloads"):
            compare_runs(*self.directories)

    def test_changed_controls_and_incomplete_runs_are_rejected(self):
        path = self.directories[1] / "run.json"
        original = json.loads(path.read_text(encoding="utf-8"))

        for mutate in (
            lambda run: run.update(status="failed"),
            lambda run: run["server"].update(expected_prefix_caching=False),
            lambda run: run["client"].update(git_revision="b" * 40),
            lambda run: run["client"]["packages"].update(httpx="changed"),
            lambda run: run["scenarios"][0]["batches"].pop(),
            lambda run: run["scenarios"][0]["batches"][0]["results"][0].update(
                ttft_ms=float("nan")
            ),
        ):
            changed = copy.deepcopy(original)
            mutate(changed)
            path.write_text(json.dumps(changed), encoding="utf-8")

            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                compare_runs(*self.directories)

    def test_reused_server_process_is_rejected_even_with_consistent_snapshots(self):
        self.edit_run(
            1, lambda run: run["server"].update(process_start_time_seconds=100)
        )

        for path in self.directories[1].glob("*.prom"):
            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    "process_start_time_seconds 101", "process_start_time_seconds 100"
                ),
                encoding="utf-8",
            )

        with self.assertRaisesRegex(ValueError, "different server processes"):
            compare_runs(*self.directories)

    def test_order_and_priming_timestamps_are_validated(self):
        path = self.directories[1] / "run.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        self.edit_run(1, lambda run: run.update(started_at="2026-01-01T00:00:00+00:00"))

        with self.assertRaisesRegex(ValueError, "required off/on"):
            compare_runs(*self.directories)

        write_json(path, original)
        self.edit_run(
            1,
            lambda run: run["scenarios"][1]["priming"].update(ended_at=run["ended_at"]),
        )

        with self.assertRaisesRegex(ValueError, "Priming"):
            compare_runs(*self.directories)

    def test_raw_metric_disagreement_is_rejected(self):
        self.edit_run(
            1, lambda run: run["scenarios"][0]["metrics"].update(new_prefill_tokens=0)
        )

        with self.assertRaisesRegex(ValueError, "raw snapshots"):
            compare_runs(*self.directories)

    def test_completed_request_contamination_is_rejected_from_raw_snapshots(self):
        path = self.directories[0] / "unique-c1-after.prom"
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace('finished_reason="stop"} 12', 'finished_reason="stop"} 13'),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Completed-request counters disagree"):
            compare_runs(*self.directories)

    def test_request_ids_and_duration_must_match_the_saved_work(self):
        path = self.directories[0] / "run.json"
        original = json.loads(path.read_text(encoding="utf-8"))

        for mutate in (
            lambda run: run["scenarios"][0]["batches"][0].update(request_ids=["wrong"]),
            lambda run: run["scenarios"][0].update(elapsed_s=0.01),
        ):
            changed = copy.deepcopy(original)
            mutate(changed)
            write_json(path, changed)

            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                compare_runs(*self.directories)

    def test_exports_include_both_rounds_and_three_readable_figures(self):
        comparison = compare_runs(
            *self.directories,
            require_cache_metrics=True,
        )
        directory = self.root / "comparison"
        arguments = [
            sys.executable,
            "-m",
            "bench.compare_cache",
            "--require-cache-metrics",
        ]

        for flag, path in zip(
            ("--round-1-off", "--round-1-on", "--round-2-off", "--round-2-on"),
            self.directories,
        ):
            arguments.extend((flag, str(path)))

        completed = subprocess.run(
            [*arguments, "--output-dir", str(directory)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Round 2", completed.stdout)
        self.assertIn("failures 0/4", completed.stdout)

        with (directory / "comparison.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 16)

        for phase in TIMING_METRICS:
            for suffix in ("mean_ms", "observations"):
                field = f"server_{phase}_{suffix}"
                self.assertTrue(all(row[field] is None for row in comparison["rows"]))
                self.assertTrue(all(row[field] == "" for row in rows))

        self.assertEqual(
            json.loads(rows[0]["finish_reasons"]),
            {
                "stop": 4,
            },
        )
        self.assertEqual(
            json.loads((directory / "comparison.json").read_text(encoding="utf-8")),
            comparison,
        )

        for name in ("01_cache_latency", "02_cache_prefill", "03_server_timings"):
            self.assertEqual(
                (directory / (name + ".png")).read_bytes()[:8], b"\x89PNG\r\n\x1a\n"
            )
            self.assertIn(
                "<svg", (directory / (name + ".svg")).read_text(encoding="utf-8")
            )

        self.assertEqual(
            (directory / "03_server_timings.svg")
            .read_text(encoding="utf-8")
            .count("N/A"),
            65,
        )

        with self.assertRaises(FileExistsError):
            export_comparison(comparison, directory)

    def test_timing_figure_distinguishes_zero_missing_and_positive_means(self):
        comparison = compare_runs(*self.directories)
        comparison["rows"][0].update(
            server_queue_mean_ms=0,
            server_queue_observations=4,
            server_prefill_mean_ms=12.5,
            server_prefill_observations=4,
            server_time_per_output_token_mean_ms=3.5,
            server_time_per_output_token_observations=4,
        )
        comparison["rows"][12].update(
            server_prefill_mean_ms=100,
            server_prefill_observations=4,
        )

        with patch("matplotlib.figure.Figure.savefig", autospec=True) as save:
            export_comparison(comparison, self.root / "partial-timings")

        figure = next(
            call.args[0]
            for call in save.call_args_list
            if call.args[1].name == "03_server_timings.png"
        )
        queue, prefill, decode, tpot = figure.axes[:4]
        self.assertEqual(queue.containers[0][0].get_height(), 0)
        self.assertEqual(sum(text.get_text() == "N/A" for text in queue.texts), 7)
        self.assertEqual(prefill.containers[0][0].get_height(), 12.5)
        self.assertEqual(tpot.containers[0][0].get_height(), 3.5)
        self.assertEqual(sum(text.get_text() == "N/A" for text in decode.texts), 8)
        self.assertEqual(tpot.get_ylabel(), "Mean TPOT (ms/token)")
        self.assertEqual(prefill.get_ylim(), figure.axes[5].get_ylim())
        self.assertGreater(prefill.get_ylim()[1], 100)

    def test_server_timing_exports_preserve_existing_metrics_and_saved_runs(self):
        baseline = compare_runs(*self.directories)
        saved_runs = [
            (directory / "run.json").read_bytes() for directory in self.directories
        ]

        for directory in self.directories:
            run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
            scale = 0.5 if run["server"]["expected_prefix_caching"] else 1

            for index, scenario in enumerate(run["scenarios"]):
                count_before = 8 + index * 10
                count_after = count_before + sum(
                    len(batch["results"]) for batch in scenario["batches"]
                )

                for key, count in (
                    ("before_snapshot", count_before),
                    ("after_snapshot", count_after),
                ):
                    path = directory / (scenario[key] + ".prom")
                    timings = "".join(
                        histogram_text(
                            seconds=count * phase_index * 0.1 * scale,
                            count=count,
                            metric=metric,
                        )
                        for phase_index, metric in enumerate(
                            TIMING_METRICS.values(), start=1
                        )
                    )
                    path.write_text(
                        path.read_text(encoding="utf-8") + timings, encoding="utf-8"
                    )

        comparison = compare_runs(*self.directories)
        directory = self.root / "timing-comparison"
        export_comparison(comparison, directory)
        exported = json.loads(
            (directory / "comparison.json").read_text(encoding="utf-8")
        )

        with (directory / "comparison.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(exported, comparison)
        self.assertEqual(comparison["changes"], baseline["changes"])

        for before, after, csv_row in zip(baseline["rows"], exported["rows"], rows):
            for field, value in before.items():
                if not field.startswith("server_"):
                    self.assertEqual(after[field], value)

            scale = 0.5 if after["cache"] == "on" else 1

            for phase_index, phase in enumerate(TIMING_METRICS, start=1):
                mean = f"server_{phase}_mean_ms"
                count = f"server_{phase}_observations"
                self.assertAlmostEqual(after[mean], phase_index * 100 * scale)
                self.assertAlmostEqual(float(csv_row[mean]), after[mean])
                self.assertEqual(after[count], 4)
                self.assertEqual(int(csv_row[count]), 4)

        self.assertEqual(
            [(directory / "run.json").read_bytes() for directory in self.directories],
            saved_runs,
        )

    def test_comparison_rejects_invalid_server_timing_windows(self):
        directory = self.directories[0]
        metric = TIMING_METRICS["queue"]

        for suffix, count in (("before", 8), ("after", 7)):
            path = directory / f"unique-c1-{suffix}.prom"
            path.write_text(
                path.read_text(encoding="utf-8")
                + histogram_text(seconds=1, count=count, metric=metric),
                encoding="utf-8",
            )

        with self.assertRaisesRegex(ValueError, "Counter decreased"):
            compare_runs(*self.directories)

    def test_zero_latency_baseline_has_no_percentage_change(self):
        self.edit_run(
            0,
            lambda run: [
                request.update(ttft_ms=0)
                for batch in run["scenarios"][0]["batches"]
                for request in batch["results"]
            ],
        )
        comparison = compare_runs(*self.directories)

        self.assertIsNone(
            comparison["changes"][0]["cache_on_change_pct"]["ttft_p50_ms"]
        )

    def test_archived_gpu_declarations_are_independent_of_current_deployment(self):
        for index in range(4):
            self.edit_run(
                index, lambda run: run["server"].update(gpu_declared="archived-gpu")
            )
        self.assertEqual(len(compare_runs(*self.directories)["rows"]), 16)

        self.edit_run(1, lambda run: run["server"].update(gpu_declared="different-gpu"))
        with self.assertRaisesRegex(ValueError, "changed between sessions"):
            compare_runs(*self.directories)

    def test_mismatched_or_missing_server_versions_are_rejected(self):
        self.edit_run(
            1,
            lambda run: run["server"].update(vllm_version_declared="different-version"),
        )
        with self.assertRaisesRegex(ValueError, "changed between sessions"):
            compare_runs(*self.directories)
        self.edit_run(1, lambda run: run["server"].update(vllm_version_declared=None))
        with self.assertRaisesRegex(ValueError, "are required"):
            compare_runs(*self.directories)

    def test_reused_round_seed_is_rejected(self):
        for index in (2, 3):
            path = self.directories[index] / "workload.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["seed"] = 42
            manifest["sha256"] = manifest_digest(manifest)
            write_json(path, manifest)
            self.edit_run(
                index, lambda run: run.update(manifest_sha256=manifest["sha256"])
            )

        with self.assertRaisesRegex(ValueError, "different workload seeds"):
            compare_runs(*self.directories)

    def test_missing_telemetry_remains_unavailable(self):
        directory = self.directories[1]

        for path in directory.glob("*.prom"):
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text(
                "\n".join(line for line in lines if "prefix_cache_hits" not in line)
                + "\n",
                encoding="utf-8",
            )

        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))

        for scenario in run["scenarios"]:
            before = parse_metrics(
                (directory / (scenario["before_snapshot"] + ".prom")).read_text(
                    encoding="utf-8"
                )
            )
            after = parse_metrics(
                (directory / (scenario["after_snapshot"] + ".prom")).read_text(
                    encoding="utf-8"
                )
            )
            scenario["metrics"] = interval_summary(before, after)

        write_json(directory / "run.json", run)
        _, _, summaries = load_run(directory, True)

        self.assertTrue(
            all(
                row["prefix_hits_tokens"] is None
                and row["token_cache_hit_rate"] is None
                for row in summaries
            )
        )

        comparison = compare_runs(*self.directories)
        output = self.root / "missing-telemetry"
        export_comparison(comparison, output)
        self.assertGreaterEqual(
            (output / "02_cache_prefill.svg").read_text(encoding="utf-8").count("N/A"),
            4,
        )

        with self.assertRaisesRegex(ValueError, "prefix_hits_tokens"):
            compare_runs(*self.directories, require_cache_metrics=True)

    def test_strict_analysis_requires_each_cache_counter(self):
        directory = self.directories[1]
        snapshots = {
            path: path.read_text(encoding="utf-8") for path in directory.glob("*.prom")
        }
        original = json.loads((directory / "run.json").read_text(encoding="utf-8"))

        for metric, field in (
            ("vllm:prefix_cache_queries_total", "prefix_queries_tokens"),
            ("vllm:prefix_cache_hits_total", "prefix_hits_tokens"),
            ("vllm:request_prefill_kv_computed_tokens_sum", "new_prefill_tokens"),
        ):
            with self.subTest(metric=metric):
                for path, text in snapshots.items():
                    path.write_text(
                        "\n".join(
                            line
                            for line in text.splitlines()
                            if not line.startswith(metric + "{")
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                run = copy.deepcopy(original)
                for scenario in run["scenarios"]:
                    before = parse_metrics(
                        (directory / (scenario["before_snapshot"] + ".prom")).read_text(
                            encoding="utf-8"
                        )
                    )
                    after = parse_metrics(
                        (directory / (scenario["after_snapshot"] + ".prom")).read_text(
                            encoding="utf-8"
                        )
                    )
                    scenario["metrics"] = interval_summary(before, after)
                write_json(directory / "run.json", run)

                comparison = compare_runs(*self.directories)
                self.assertTrue(
                    all(
                        row[field] is None
                        for row in comparison["rows"]
                        if row["round"] == 1 and row["cache"] == "on"
                    )
                )
                with self.assertRaisesRegex(ValueError, field):
                    compare_runs(*self.directories, require_cache_metrics=True)

    def test_real_runner_outputs_are_accepted_without_schema_translation(self):
        async def execute():
            for session, index in enumerate((0, 1, 3, 2)):
                expected = index in (1, 3)
                manifest = create_manifest(
                    WordTokenizer(), seed=42 if index < 2 else 43, count=2
                )
                server = StudyServer(manifest, expected=expected)

                def transport(request):
                    response = server(request)

                    if request.url.path == "/metrics":
                        return httpx.Response(
                            200,
                            text=response.text.replace(
                                "process_start_time_seconds 100",
                                f"process_start_time_seconds {100 + session}",
                            ),
                        )

                    return response

                log = self.root / f"runner-{session}.log"
                log.write_text(
                    "Starting vLLM server with command: "
                    + " ".join(ServerConfig(prefix_caching=expected).command()),
                    encoding="utf-8",
                )

                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(transport), base_url="http://lab"
                ) as client:
                    await run_study(
                        client,
                        manifest,
                        directory=self.root / f"runner-{index}",
                        expected=expected,
                        server_log=log,
                    )

        with patch("bench.async_client.VLLM_BASE_URL", "http://lab"):
            asyncio.run(execute())

        comparison = compare_runs(
            *(self.root / f"runner-{index}" for index in range(4))
        )
        self.assertEqual(len(comparison["rows"]), 16)
        self.assertTrue(
            all(
                row["attempted"] == 2 and row["failed"] == 0
                for row in comparison["rows"]
            )
        )


if __name__ == "__main__":
    unittest.main()
