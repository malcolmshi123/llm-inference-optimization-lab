import os
import httpx
import json
import time

from bench.async_client import StreamingResponseError

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "")
MODAL_PROXY_KEY = os.environ.get("MODAL_PROXY_KEY", "")
MODAL_PROXY_SECRET = os.environ.get("MODAL_PROXY_SECRET", "")

HEADERS = {
    "Modal-Key": MODAL_PROXY_KEY,
    "Modal-Secret": MODAL_PROXY_SECRET,
    "Content-Type": "application/json",
}

def send_streaming_request():
    if not all((VLLM_BASE_URL, MODAL_PROXY_KEY, MODAL_PROXY_SECRET)):
        raise ValueError(
            "Set VLLM_BASE_URL, MODAL_PROXY_KEY, and MODAL_PROXY_SECRET "
            "before running the smoke test."
        )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": "Explain what a KV cache is in one sentence.",
            }
        ],
        "max_tokens": 64,
        "temperature": 0,
        "stream": True,
    }

    with httpx.Client(timeout=600, follow_redirects=True) as client:
        first_token_time = None
        received_done = False
        finish_reason = None
        start_time = time.perf_counter()

        with client.stream(
            "POST",
            f"{VLLM_BASE_URL}/v1/chat/completions",
            json=payload,
            headers=HEADERS,
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    received_done = True
                    break
                event = json.loads(data)
                if not isinstance(event, dict):
                    raise StreamingResponseError("Expected a JSON object in SSE data.")
                if "error" in event:
                    raise StreamingResponseError(
                        f"Server streamed an error: {event['error']}"
                    )
                if not event.get("choices"):
                    continue
                choice = event["choices"][0]
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                content = choice.get("delta", {}).get("content")
                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    print(content, end="", flush=True)

            end_time = time.perf_counter()

            if (
                not received_done
                or finish_reason not in {"stop", "length"}
                or first_token_time is None
            ):
                raise StreamingResponseError(
                    "Stream did not contain generated text, a successful finish reason, and [DONE]."
                )

            print()
            print("Status: ", response.status_code)

        if first_token_time is not None:
            ttft_ms = (first_token_time - start_time) * 1000
            print(f"Time to first token: {ttft_ms:.2f} ms")
        else:
            print("No tokens received.")

        e2e_time_ms = (end_time - start_time) * 1000
        print(f"End-to-end time: {e2e_time_ms:.2f} ms")

if __name__ == "__main__":
    send_streaming_request()
