import os
import httpx
import json
import time

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

VLLM_BASE_URL = os.environ["VLLM_BASE_URL"]
MODAL_PROXY_KEY = os.environ["MODAL_PROXY_KEY"]
MODAL_PROXY_SECRET = os.environ["MODAL_PROXY_SECRET"]

HEADERS = {
    "Modal-Key": MODAL_PROXY_KEY,
    "Modal-Secret": MODAL_PROXY_SECRET,
    "Content-Type": "application/json",
}

def send_streaming_request():
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
        "stream": True
    }

    with httpx.Client(timeout=600, follow_redirects=True) as client:
        first_token_time = None
        start_time = time.perf_counter()

        with client.stream("POST", f"{VLLM_BASE_URL}/v1/chat/completions", json=payload, headers=HEADERS) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    content = event["choices"][0]["delta"].get("content")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        print(content, end="", flush=True)

            end_time = time.perf_counter()

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