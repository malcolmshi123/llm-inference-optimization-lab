import asyncio
import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import httpx

from bench.cache_study import run_study, startup_command, write_text
from bench.cache_workloads import create_manifest, token_ids
from server_config import ServerConfig
from tests.test_cache_metrics import metric_text
from tests.test_cache_workloads import WordTokenizer

class StudyServer:
    def __init__(
        self,
        manifest,
        *,
        expected=True,
        fail_request=None,
        restart=False,
        cancel_request=None,
        mismatch=False,
        busy_reads=0,
    ):
        self.expected = expected
        self.fail_request = fail_request
        self.restart = restart
        self.cancel_request = cancel_request
        self.mismatch = mismatch
        self.busy_reads = busy_reads
        self.requests = []
        self.tokens = 0
        self.hits = 0
        self.measured = 0
        self.completed = 0
        self.cached = set()
        self.measured_requests = {
            request["messages"][0]["content"]: request
            for scenario in manifest["scenarios"]
            for request in scenario["requests"]
        }

    def __call__(self, request):
        if request.url.path == "/metrics":
            text = metric_text(
                tokens=self.tokens,
                hits=self.hits,
                prefill=self.tokens - self.hits,
                identity=101 if self.restart and self.measured else 100,
                expected=self.expected,
                finished={
                    "stop": self.completed,
                },
            )

            if self.busy_reads:
                self.busy_reads -= 1
                text = (
                    "\n".join(
                        line[:-1] + "1"
                        if line.startswith("vllm:num_requests_running")
                        else line
                        for line in text.splitlines()
                    )
                    + "\n"
                )

            return httpx.Response(200, text=text)

        payload = json.loads(request.content)
        content = payload["messages"][0]["content"]
        tokens = len(token_ids(WordTokenizer(), payload["messages"]))
        source = self.measured_requests.get(content)
        prefix = content.splitlines()[0]
        hits = (
            source["shared_prefix_tokens"]
            if source and self.expected and prefix in self.cached
            else 0
        )
        self.tokens += tokens
        self.hits += hits or 0
        self.cached.add(prefix)
        self.requests.append(payload)

        if source:
            self.measured += 1

            if self.measured == self.cancel_request:
                raise asyncio.CancelledError()

            if self.measured == self.fail_request:
                return httpx.Response(
                    200, text='data: {"error": {"message": "test failure"}}\n\n'
                )

        self.completed += 1
        events = [
            {
                "choices": [
                    {
                        "delta": {
                            "content": "Answer",
                        },
                    }
                ],
            },
            {
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            },
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": tokens + (1 if source and self.mismatch else 0),
                    "completion_tokens": 1,
                },
            },
        ]
        body = (
            "".join("data: " + json.dumps(event) + "\n\n" for event in events)
            + "data: [DONE]\n\n"
        )

        return httpx.Response(200, text=body)

class CacheStudyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = create_manifest(WordTokenizer(), count=2)
        self.log = self.root / "server.log"
        self.log.write_text(
            "Starting vLLM server with command: " + " ".join(ServerConfig().command()),
            encoding="utf-8",
        )
        self.patch = patch("bench.async_client.VLLM_BASE_URL", "http://lab")
        self.patch.start()
        self.addCleanup(self.patch.stop)

    async def execute(self, server, name="session", **options):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(server), base_url="http://lab"
        ) as client:
            return await run_study(
                client,
                self.manifest,
                directory=self.root / name,
                expected=True,
                server_log=self.log,
                **options,
            )

    async def test_preparation_is_excluded_and_saved_payloads_are_replayed(self):
        server = StudyServer(self.manifest)
        run = await self.execute(server)

        self.assertEqual(run["status"], "completed")
        self.assertEqual(server.measured, 8)
        self.assertEqual(len(server.requests), 18)

        for source, scenario in zip(self.manifest["scenarios"], run["scenarios"]):
            self.assertEqual(scenario["status"], "completed")
            self.assertGreater(scenario["elapsed_s"], 0)
            expected_tokens = sum(
                request["prompt_tokens"] for request in source["requests"]
            )
            self.assertEqual(
                scenario["metrics"]["prefix_queries_tokens"], expected_tokens
            )
            self.assertEqual(
                sum(len(batch["results"]) for batch in scenario["batches"]), 2
            )

            if source["case"] == "shared":
                self.assertGreater(scenario["metrics"]["token_cache_hit_rate"], 0.85)
            else:
                self.assertEqual(scenario["metrics"]["token_cache_hit_rate"], 0)

        saved = json.loads(
            (self.root / "session" / "workload.json").read_text(encoding="utf-8")
        )
        self.assertEqual(saved, self.manifest)

        with (self.root / "session" / "requests.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 8)
        self.assertTrue(
            all(
                row["finish_reason"] == "stop" and row["completion_tokens"] == "1"
                for row in rows
            )
        )

    async def test_streaming_failure_retains_attempts_metrics_and_partial_results(self):
        server = StudyServer(self.manifest, fail_request=2)

        with self.assertRaisesRegex(ValueError, "Measured requests failed"):
            await self.execute(server)

        saved = json.loads(
            (self.root / "session" / "run.json").read_text(encoding="utf-8")
        )
        scenario = saved["scenarios"][0]

        self.assertEqual(saved["status"], "failed")
        self.assertEqual(scenario["status"], "failed")
        self.assertEqual(len(scenario["batches"]), 2)
        self.assertFalse(scenario["batches"][-1]["results"][0]["success"])
        self.assertIn("metrics", scenario)
        self.assertEqual(len(saved["scenarios"]), 1)

    async def test_extra_server_request_fails_and_retains_measured_results(self):
        server = StudyServer(self.manifest)

        def contaminated(request):
            response = server(request)
            if request.url.path == "/metrics" and server.measured:
                lines = response.text.splitlines()
                lines = [
                    line.rsplit(" ", 1)[0] + f" {server.completed + 1}"
                    if line.startswith("vllm:request_success_total{")
                    else line
                    for line in lines
                ]
                return httpx.Response(200, text="\n".join(lines) + "\n")
            return response

        with self.assertRaisesRegex(ValueError, "Completed-request counters disagree"):
            await self.execute(contaminated)

        saved = json.loads(
            (self.root / "session" / "run.json").read_text(encoding="utf-8")
        )
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["scenarios"][0]["status"], "failed")
        self.assertEqual(len(saved["scenarios"][0]["batches"]), 2)
        self.assertIn("unique-c1-after", saved["snapshots"])

    async def test_wrong_running_treatment_stops_before_warmup(self):
        server = StudyServer(self.manifest, expected=False)

        with self.assertRaisesRegex(ValueError, "treatment"):
            await self.execute(server)

        self.assertEqual(server.requests, [])
        self.assertEqual(
            json.loads((self.root / "session" / "run.json").read_text())["status"],
            "failed",
        )

    async def test_restart_and_reused_previous_session_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "restarted"):
            await self.execute(StudyServer(self.manifest, restart=True), name="restart")

        previous = self.root / "previous.json"
        previous.write_text(
            json.dumps(
                {
                    "kind": "cache_study",
                    "server": {
                        "process_start_time_seconds": 100,
                    },
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "fresh server"):
            await self.execute(StudyServer(self.manifest), previous_run=previous)

    async def test_existing_output_is_not_overwritten(self):
        await self.execute(StudyServer(self.manifest))

        with self.assertRaises(FileExistsError):
            await self.execute(StudyServer(self.manifest))

    async def test_cancellation_retains_completed_batches(self):
        with self.assertRaises(asyncio.CancelledError):
            await self.execute(StudyServer(self.manifest, cancel_request=2))

        saved = json.loads(
            (self.root / "session" / "run.json").read_text(encoding="utf-8")
        )

        self.assertEqual(saved["status"], "interrupted")
        self.assertEqual(len(saved["scenarios"][0]["batches"]), 1)

    async def test_server_token_count_mismatch_is_a_recorded_failure(self):
        with self.assertRaisesRegex(ValueError, "Measured requests failed"):
            await self.execute(StudyServer(self.manifest, mismatch=True))

        saved = json.loads(
            (self.root / "session" / "run.json").read_text(encoding="utf-8")
        )

        self.assertEqual(
            saved["scenarios"][0]["batches"][0]["results"][0]["error_type"],
            "TokenCountMismatch",
        )

    async def test_idle_polling_waits_before_warmup(self):
        with patch("bench.cache_study.asyncio.sleep", new_callable=AsyncMock) as sleep:
            run = await self.execute(StudyServer(self.manifest, busy_reads=1))

        self.assertEqual(run["status"], "completed")
        sleep.assert_awaited_once_with(1)

    def test_startup_log_uses_the_last_command_and_checks_fixed_controls(self):
        baseline = "Starting vLLM server with command: " + " ".join(
            ServerConfig().command()
        )

        self.assertEqual(startup_command(baseline, True), ServerConfig().command())

        for log in (
            "no command",
            baseline.replace("2048", "4096"),
            baseline
            + "\n"
            + baseline.replace("--enable-prefix-caching", "--no-enable-prefix-caching"),
        ):
            with self.subTest(log=log), self.assertRaises(ValueError):
                startup_command(log, True)

    def test_transient_replacement_lock_is_retried(self):
        path = self.root / "requests.csv"
        path.write_text("old", encoding="utf-8")
        replace = Path.replace
        attempts = 0

        def locked_once(source, destination):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise PermissionError("temporary file lock")
            return replace(source, destination)

        with (
            patch.object(Path, "replace", locked_once),
            patch("bench.cache_study.time.sleep") as sleep,
        ):
            write_text(path, "new")

        self.assertEqual(path.read_text(encoding="utf-8"), "new")
        self.assertFalse(path.with_suffix(".csv.tmp").exists())
        sleep.assert_called_once_with(0.05)

    def test_persistent_replacement_lock_fails_without_overwriting_results(self):
        path = self.root / "requests.csv"
        path.write_text("old", encoding="utf-8")

        with (
            patch.object(Path, "replace", side_effect=PermissionError("locked")),
            patch("bench.cache_study.time.sleep") as sleep,
        ):
            with self.assertRaises(PermissionError):
                write_text(path, "new")

        self.assertEqual(path.read_text(encoding="utf-8"), "old")
        self.assertEqual(
            path.with_suffix(".csv.tmp").read_text(encoding="utf-8"), "new"
        )
        self.assertEqual(sleep.call_count, 4)

if __name__ == "__main__":
    unittest.main()
