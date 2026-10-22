import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx

from bench.cache_workloads import create_manifest
from bench.profile_capture import capture
from server_config import ServerConfig
from tests.test_cache_study import StudyServer
from tests.test_cache_workloads import WordTokenizer

class ProfilingServer(StudyServer):
    def __init__(self, manifest, *, start_timeout=False, stop_status=200, **options):
        super().__init__(manifest, **options)
        self.active = False
        self.events = []
        self.profiled_payloads = []
        self.start_timeout = start_timeout
        self.stop_status = stop_status

    def __call__(self, request):
        path = request.url.path

        if path == "/start_profile":
            self.events.append("start")
            self.active = True
            if self.start_timeout:
                raise httpx.ReadTimeout("Start response was lost", request=request)
            return httpx.Response(200)

        if path == "/stop_profile":
            self.events.append("stop")
            self.active = False
            return httpx.Response(self.stop_status)

        if path == "/metrics":
            self.events.append("metrics")
        else:
            payload = json.loads(request.content)
            self.events.append("captured" if self.active else "preparation")
            if self.active:
                self.profiled_payloads.append(payload)

        return super().__call__(request)

class ProfileCaptureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = create_manifest(WordTokenizer(), count=10)
        self.log = self.root / "server.log"
        for target, value in (
            ("bench.async_client.VLLM_BASE_URL", "http://lab"),
            (
                "bench.profile_capture.client_metadata",
                lambda: {
                    "git_revision": "test",
                },
            ),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def execute(
        self, server, *, name="capture", expected=True, log_session="test"
    ):
        self.log.write_text(
            "Starting vLLM server with command: "
            + " ".join(
                ServerConfig(
                    prefix_caching=expected, profile_session=log_session
                ).command()
            ),
            encoding="utf-8",
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(server), base_url="http://lab"
        ) as client:
            return await capture(
                client,
                self.manifest,
                directory=self.root / name,
                expected=expected,
                profile_session="test",
                server_log=self.log,
            )

    def saved(self):
        return json.loads(
            (self.root / "capture/capture.json").read_text(encoding="utf-8")
        )

    async def test_only_first_eight_shared_requests_are_profiled_for_each_treatment(
        self,
    ):
        source = self.manifest["scenarios"][-1]
        for expected in (False, True):
            with self.subTest(expected=expected):
                server = ProfilingServer(self.manifest, expected=expected)
                name = "on" if expected else "off"
                run = await self.execute(server, name=name, expected=expected)
                self.assertEqual(run["kind"], "profile_capture")
                self.assertEqual(run["status"], "completed")
                self.assertTrue(run["profiler"]["stop_completed"])
                self.assertFalse(run["profiler"]["trace_files_verified"])
                self.assertEqual(run["profiler"]["directory"], "/profiles/test")
                self.assertEqual(
                    run["request_ids"],
                    [r["request_id"] for r in source["requests"][:8]],
                )
                self.assertEqual(
                    [r["messages"] for r in server.profiled_payloads],
                    [r["messages"] for r in source["requests"][:8]],
                )
                start, stop = server.events.index("start"), server.events.index("stop")
                self.assertEqual(server.events[start + 1 : stop], ["captured"] * 8)
                self.assertEqual(server.events[:start].count("preparation"), 9)
                self.assertEqual(len(run["warmup"]["results"]), 8)
                self.assertEqual(run["priming"]["max_tokens"], 1)
                self.assertEqual(run["server"]["process_start_time_seconds"], 100)
                if expected:
                    self.assertGreater(run["metrics"]["prefix_hits_tokens"], 0)
                else:
                    self.assertEqual(run["metrics"]["prefix_hits_tokens"], 0)
                self.assertEqual(
                    json.loads(
                        (self.root / name / "workload.json").read_text(encoding="utf-8")
                    ),
                    self.manifest,
                )

    async def test_failed_requests_are_saved_and_profiler_is_stopped(self):
        server = ProfilingServer(self.manifest, fail_request=3)
        with self.assertRaisesRegex(ValueError, "captured_batch requests failed"):
            await self.execute(server)
        run = self.saved()
        self.assertEqual(run["status"], "failed")
        self.assertEqual(
            sum(not r["success"] for r in run["captured_batch"]["results"]), 1
        )
        self.assertTrue(run["profiler"]["stop_completed"])

    async def test_token_mismatches_are_failures_and_keep_results(self):
        with self.assertRaisesRegex(ValueError, "captured_batch requests failed"):
            await self.execute(ProfilingServer(self.manifest, mismatch=True))
        run = self.saved()
        self.assertTrue(run["profiler"]["stop_completed"])
        self.assertTrue(
            all(
                r["error_type"] == "TokenCountMismatch"
                for r in run["captured_batch"]["results"]
            )
        )

    async def test_start_timeout_still_attempts_stop_without_sending_capture_batch(
        self,
    ):
        server = ProfilingServer(self.manifest, start_timeout=True)
        with self.assertRaises(httpx.ReadTimeout):
            await self.execute(server)
        run = self.saved()
        self.assertFalse(run["profiler"]["start_completed"])
        self.assertTrue(run["profiler"]["stop_completed"])
        self.assertNotIn("captured_batch", run)
        self.assertEqual(server.profiled_payloads, [])

    async def test_stop_failure_does_not_hide_the_original_request_failure(self):
        server = ProfilingServer(self.manifest, fail_request=3, stop_status=500)
        with self.assertRaisesRegex(ValueError, "captured_batch requests failed"):
            await self.execute(server)
        run = self.saved()
        self.assertEqual(run["error_type"], "ValueError")
        self.assertEqual(run["profiler"]["stop_error"]["type"], "HTTPStatusError")
        self.assertFalse(run["profiler"]["stop_completed"])

    async def test_stop_failure_marks_an_otherwise_successful_capture_failed(self):
        with self.assertRaises(httpx.HTTPStatusError):
            await self.execute(ProfilingServer(self.manifest, stop_status=500))
        self.assertEqual(self.saved()["status"], "failed")

    async def test_cancellation_attempts_cleanup_and_records_interruption(self):
        with self.assertRaises(asyncio.CancelledError):
            await self.execute(ProfilingServer(self.manifest, cancel_request=3))
        run = self.saved()
        self.assertEqual(run["status"], "interrupted")
        self.assertTrue(run["profiler"]["stop_completed"])

    async def test_wrong_treatment_is_rejected_before_preparation_or_profiling(self):
        server = ProfilingServer(self.manifest, expected=False)
        with self.assertRaises(ValueError):
            await self.execute(server)
        self.assertEqual(server.events, ["metrics"])
        self.assertFalse(self.saved()["profiler"]["start_attempted"])

    async def test_restart_invalidates_capture_after_profiler_is_stopped(self):
        with self.assertRaisesRegex(ValueError, "restarted"):
            await self.execute(ProfilingServer(self.manifest, restart=True))
        self.assertEqual(self.saved()["status"], "failed")
        self.assertTrue(self.saved()["profiler"]["stop_completed"])

    async def test_log_mismatch_existing_outputs_and_small_workloads_send_no_requests(
        self,
    ):
        server = ProfilingServer(self.manifest)
        with self.assertRaisesRegex(ValueError, "Startup log"):
            await self.execute(server, log_session="different")
        self.assertFalse((self.root / "capture").exists())
        (self.root / "capture").mkdir()
        (self.root / "capture/keep.txt").write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            await self.execute(server)
        self.assertEqual(
            (self.root / "capture/keep.txt").read_text(encoding="utf-8"), "keep"
        )
        self.manifest = create_manifest(WordTokenizer(), count=4)
        with self.assertRaisesRegex(ValueError, "eight"):
            await self.execute(server, name="small")
        self.assertEqual(server.events, [])

if __name__ == "__main__":
    unittest.main()
