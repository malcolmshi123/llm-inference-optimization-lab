import contextlib
import io
import unittest
from unittest.mock import patch

import httpx

from bench import client as smoke
from bench.async_client import StreamingResponseError
from tests.test_async_client import CONTENT, FINISH, USAGE, sse


class SmokeClientTests(unittest.TestCase):
    def execute(self, body):
        client = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text=body)
            )
        )
        output = io.StringIO()
        with (
            patch.multiple(
                smoke,
                VLLM_BASE_URL="http://lab",
                MODAL_PROXY_KEY="test",
                MODAL_PROXY_SECRET="test",
            ),
            patch.object(smoke.httpx, "Client", return_value=client),
            contextlib.redirect_stdout(output),
        ):
            smoke.send_streaming_request()
        return output.getvalue()

    def test_usage_only_event_does_not_break_smoke_test(self):
        output = self.execute(sse(CONTENT, FINISH, USAGE, "[DONE]"))
        self.assertIn("Hello", output)
        self.assertIn("Time to first token:", output)

    def test_partial_and_error_streams_are_rejected(self):
        for body in (
            sse(CONTENT),
            sse(CONTENT, FINISH, USAGE),
            sse(
                {
                    "error": {
                        "message": "failed",
                    },
                }
            ),
        ):
            with self.subTest(body=body), self.assertRaises(StreamingResponseError):
                self.execute(body)

    def test_missing_credentials_are_reported_before_request(self):
        with (
            patch.object(smoke, "VLLM_BASE_URL", ""),
            self.assertRaisesRegex(ValueError, "VLLM_BASE_URL"),
        ):
            smoke.send_streaming_request()


if __name__ == "__main__":
    unittest.main()
