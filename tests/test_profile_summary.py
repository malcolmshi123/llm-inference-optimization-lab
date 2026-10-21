import copy
import csv
import gzip
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bench.cache_study import write_json
from bench.cache_workloads import create_manifest
from bench.profile_summary import load_capture, summarize_pair, summarize_trace
from server_config import ServerConfig
from tests.test_cache_workloads import WordTokenizer
from tests.test_compare_cache import create_run


def kernel(name, start, duration, device=0):
    return {
        "cat": "kernel",
        "ph": "X",
        "name": name,
        "ts": start,
        "dur": duration,
        "args": {
            "device": device,
        },
    }


def annotation(name, start=900, duration=50):
    return {
        "cat": "user_annotation",
        "ph": "X",
        "name": name,
        "ts": start,
        "dur": duration,
    }


def write_traces(directory, trace):
    traces = directory / "traces"
    traces.mkdir(exist_ok=True)
    records = []
    for name, value in (
        ("worker.json.gz", trace),
        (
            "frontend.json.gz",
            {
                "traceEvents": [],
            },
        ),
    ):
        path = traces / name
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(value, handle)
        records.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "gpu_kernel_events": sum(
                    event.get("cat") == "kernel" and event.get("dur", 0) > 0
                    for event in value["traceEvents"]
                ),
            }
        )
    write_json(
        directory / "trace-check.json",
        {
            "kind": "profile_trace_check",
            "traces": records,
        },
    )


def fixture(directory, manifest, *, expected, session):
    create_run(directory, manifest, expected=expected, session=session)
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    scenario = run["scenarios"][-1]
    config = ServerConfig(prefix_caching=expected, profile_session=directory.name)
    record = {
        "schema_version": 1,
        "kind": "profile_capture",
        "status": "completed",
        "scenario": "shared-c8",
        "server": {
            "expected_prefix_caching": expected,
            "process_start_time_seconds": run["server"]["process_start_time_seconds"],
            "startup_command_from_log": config.command(),
        },
        "profiler": {
            "session": directory.name,
            "start_completed": True,
            "stop_completed": True,
        },
        "manifest_sha256": manifest["sha256"],
        "request_ids": [
            request["request_id"] for request in manifest["scenarios"][-1]["requests"]
        ],
        "captured_batch": {
            "results": scenario["batches"][0]["results"],
        },
        "metrics": scenario["metrics"],
    }
    write_json(directory / "capture.json", record)
    for phase in ("before", "after"):
        (directory / f"capture-{phase}.prom").write_bytes(
            (directory / f"shared-c8-{phase}.prom").read_bytes()
        )
    context_tokens = int(record["metrics"]["new_prefill_tokens"])
    trace = {
        "displayTimeUnit": "ms",
        "deviceProperties": [
            {
                "id": 0,
                "name": "NVIDIA A10",
            }
        ],
        "traceEvents": [
            annotation(f"execute_context_8({context_tokens})_generation_0(0)"),
            annotation("execute_context_0(0)_generation_8(8)", start=1400),
            kernel("full<kernel, template, arguments>", 1000, 300),
            kernel("full<kernel, template, arguments>", 1200, 300),
            kernel("another", 1600, 100),
        ],
    }
    write_traces(directory, trace)
    return trace


class TraceSummaryTests(unittest.TestCase):
    def test_kernel_totals_and_union_are_distinct_and_use_microseconds(self):
        result = summarize_trace(
            {
                "displayTimeUnit": "ms",
                "traceEvents": [
                    annotation("execute_context_1(2048)_generation_0(0)"),
                    kernel("full<name>", 1000, 300),
                    kernel("full<name>", 1200, 300),
                    kernel("other", 1600, 100),
                    kernel("ignored_zero", 2000, 0),
                ],
            }
        )
        self.assertEqual(result["kernel_count"], 3)
        self.assertAlmostEqual(result["summed_kernel_ms"], 0.7)
        self.assertAlmostEqual(result["kernel_union_ms"], 0.6)
        self.assertAlmostEqual(result["kernel_span_ms"], 0.7)
        self.assertEqual(
            result["kernels"][0],
            {
                "name": "full<name>",
                "calls": 2,
                "total_ms": 0.6,
                "mean_ms": 0.3,
            },
        )
        self.assertAlmostEqual(result["timeline_kernel_intervals_ms"][0][0], 0.1)
        self.assertAlmostEqual(result["timeline_kernel_intervals_ms"][0][1], 0.7)
        self.assertEqual(result["timeline_gap_ms"], 0.5)
        self.assertEqual(result["annotations"][0]["duration_ms"], 0.05)

    def test_cpu_annotations_are_classified_without_assigning_gpu_time(self):
        result = summarize_trace(
            {
                "traceEvents": [
                    annotation("execute_context_1(224)_generation_0(0)"),
                    annotation("execute_context_2(448)_generation_3(3)"),
                    annotation("execute_context_0(0)_generation_8(8)"),
                    annotation("execute_context_0(0)_generation_0(0)"),
                ],
            }
        )
        by_name = {row["name"]: row for row in result["annotations"]}
        self.assertEqual(
            by_name["execute_context_1(224)_generation_0(0)"]["phase"], "context_only"
        )
        self.assertEqual(
            by_name["execute_context_2(448)_generation_3(3)"]["phase"], "mixed"
        )
        self.assertEqual(
            by_name["execute_context_0(0)_generation_8(8)"]["phase"], "generation_only"
        )
        self.assertEqual(
            by_name["execute_context_0(0)_generation_0(0)"]["phase"], "empty"
        )
        self.assertEqual(result["kernel_count"], 0)

    def test_invalid_durations_timestamps_devices_and_phases_are_rejected(self):
        for field, value in (
            ("dur", -1),
            ("dur", float("nan")),
            ("dur", True),
            ("ts", "100"),
            ("ts", float("inf")),
            ("ph", "B"),
        ):
            event = kernel("name", 1000, 1)
            event[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                summarize_trace(
                    {
                        "traceEvents": [event],
                    }
                )
        with self.assertRaisesRegex(ValueError, "one GPU"):
            summarize_trace(
                {
                    "traceEvents": [kernel("a", 0, 1), kernel("b", 0, 1, device=1)],
                }
            )


class ProfileSummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        manifest = create_manifest(WordTokenizer(), count=8)
        self.off, self.on = self.root / "off", self.root / "on"
        self.off_trace = fixture(self.off, manifest, expected=False, session=0)
        fixture(self.on, manifest, expected=True, session=1)

    def test_real_cli_exports_full_names_annotations_and_figures_without_rewriting_inputs(
        self,
    ):
        before = {
            path: path.read_bytes()
            for directory in (self.off, self.on)
            for path in directory.rglob("*")
            if path.is_file()
        }
        output = self.root / "comparison"
        command = [
            sys.executable,
            "-m",
            "bench.profile_summary",
            "--off",
            str(self.off),
            "--on",
            str(self.on),
            "--output-dir",
            str(output),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertIn("Cache off: 3 kernels", result.stdout)
        self.assertEqual(
            json.loads((output / "summary.json").read_text(encoding="utf-8")),
            json.loads(json.dumps(summarize_pair(self.off, self.on))),
        )
        with (output / "kernels.csv").open(encoding="utf-8", newline="") as handle:
            kernels = list(csv.DictReader(handle))
        self.assertEqual(kernels[0]["name"], "full<kernel, template, arguments>")
        self.assertEqual(kernels[0]["calls"], "2")
        with (output / "annotations.csv").open(encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 4)
        self.assertEqual(
            (output / "01_gpu_timeline.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n"
        )
        svg = (output / "01_gpu_timeline.svg").read_text(encoding="utf-8")
        self.assertIn("Host scopes", svg)
        self.assertIn("GPU kernels", svg)
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_trace_hash_changes_are_rejected(self):
        (self.off / "traces/worker.json.gz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash"):
            load_capture(self.off, False)

    def test_receipt_counts_and_paths_must_match_downloads(self):
        path = self.off / "trace-check.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for field, value in (("gpu_kernel_events", 999), ("path", "../outside.gz")):
            changed = copy.deepcopy(original)
            changed["traces"][0][field] = value
            write_json(path, changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_capture(self.off, False)

    def test_unreceipted_traces_are_rejected(self):
        (self.off / "traces/extra.json.gz").write_bytes(b"unlisted")
        with self.assertRaisesRegex(ValueError, "every downloaded"):
            load_capture(self.off, False)

    def test_annotation_context_tokens_must_reconcile_with_server_counters(self):
        self.off_trace["traceEvents"][0]["name"] = (
            "execute_context_8(1)_generation_0(0)"
        )
        write_traces(self.off, self.off_trace)
        with self.assertRaisesRegex(ValueError, "context tokens"):
            load_capture(self.off, False)

    def test_partial_captures_wrong_treatments_and_tampered_metrics_are_rejected(self):
        path = self.off / "capture.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for changed in (
            dict(original, status="failed"),
            dict(original, metrics={}),
        ):
            write_json(path, changed)
            with self.assertRaises(ValueError):
                load_capture(self.off, False)
        write_json(path, original)
        with self.assertRaises(ValueError):
            load_capture(self.off, True)

    def test_reused_server_processes_are_rejected(self):
        path = self.on / "capture.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["server"]["process_start_time_seconds"] = 100
        write_json(path, record)
        for phase in ("before", "after"):
            path = self.on / f"capture-{phase}.prom"
            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    "process_start_time_seconds 101", "process_start_time_seconds 100"
                ),
                encoding="utf-8",
            )
        with self.assertRaisesRegex(ValueError, "fresh"):
            summarize_pair(self.off, self.on)

    def test_different_workloads_are_rejected(self):
        other = self.root / "other"
        fixture(
            other,
            create_manifest(WordTokenizer(), count=8, seed=43),
            expected=True,
            session=1,
        )
        with self.assertRaisesRegex(ValueError, "same workload"):
            summarize_pair(self.off, other)


if __name__ == "__main__":
    unittest.main()
