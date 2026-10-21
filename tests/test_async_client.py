import json
import unittest

from bench.async_client import run_batch


def sse(*events):
    return "\n\n".join(
        "data: " + (event if isinstance(event, str) else json.dumps(event))
        for event in events
    ) + "\n\n"


CONTENT = {"choices": [{"delta": {"content": "Hello"}}]}
FINISH = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
USAGE = {
    "choices": [],
    "usage": {"prompt_tokens": 256, "completion_tokens": 1},
}


class FakeResponse:
    status_code = 200

    def __init__(self, body):
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        for line in self.body.splitlines():
            yield line


class FakeClient:
    def __init__(self, body):
        self.body = body

    def stream(self, *args, **kwargs):
        return FakeResponse(self.body)


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def result_for(self, body):
        batch = await run_batch(FakeClient(body), ["prompt"])
        return batch["results"][0]

    def assert_failed(self, result):
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "StreamingResponseError")
        self.assertIsNone(result["ttft_ms"])
        self.assertIsNone(result["e2e_ms"])
        self.assertIsNone(result["completion_tokens"])

    async def test_complete_streams_succeed(self):
        for reason in ("stop", "length"):
            with self.subTest(finish_reason=reason):
                finish = {"choices": [{"delta": {}, "finish_reason": reason}]}
                result = await self.result_for(
                    sse(CONTENT, finish, USAGE, "[DONE]")
                )
                self.assertTrue(result["success"])
                self.assertEqual(result["generated_text"], "Hello")
                self.assertEqual(result["finish_reason"], reason)
                self.assertEqual(result["prompt_tokens"], 256)
                self.assertEqual(result["completion_tokens"], 1)
                self.assertGreaterEqual(result["ttft_ms"], 0)
                self.assertGreaterEqual(result["e2e_ms"], result["ttft_ms"])

    async def test_incomplete_and_error_streams_fail(self):
        error = {"error": {"message": "engine failed", "type": "server_error"}}
        cases = {
            "empty body": "",
            "partial content then EOF": sse(CONTENT),
            "missing DONE": sse(CONTENT, FINISH, USAGE),
            "missing finish reason": sse(CONTENT, USAGE, "[DONE]"),
            "DONE only": sse("[DONE]"),
            "error only": sse(error),
            "error after content": sse(CONTENT, error, "[DONE]"),
            "error after finish": sse(CONTENT, FINISH, error, "[DONE]"),
            "missing usage": sse(CONTENT, FINISH, "[DONE]"),
            "no generated text": sse(FINISH, USAGE, "[DONE]"),
            "error finish reason": sse(
                CONTENT,
                {"choices": [{"delta": {}, "finish_reason": "error"}]},
                USAGE,
                "[DONE]",
            ),
        }
        for label, body in cases.items():
            with self.subTest(case=label):
                self.assert_failed(await self.result_for(body))

    async def test_invalid_token_usage_fails(self):
        for prompt_tokens, completion_tokens in (
            (None, 1), (256, None), (-1, 1), (256, -1),
            (256, 0), (True, 1), (256, True), (256, 1.5),
        ):
            with self.subTest(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            ):
                usage = {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                }
                self.assert_failed(await self.result_for(
                    sse(CONTENT, FINISH, usage, "[DONE]")
                ))


if __name__ == "__main__":
    unittest.main()
