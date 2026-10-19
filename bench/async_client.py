import os
import asyncio
import httpx
import json
import statistics
import time
from bench.workloads import generate_prompt

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

VLLM_BASE_URL = os.environ["VLLM_BASE_URL"]
MODAL_PROXY_KEY = os.environ["MODAL_PROXY_KEY"]
MODAL_PROXY_SECRET = os.environ["MODAL_PROXY_SECRET"]

HEADERS = {
    "Modal-Key": MODAL_PROXY_KEY,
    "Modal-Secret": MODAL_PROXY_SECRET,
    "Content-Type": "application/json",
}

async def send_streaming_request(client: httpx.AsyncClient, prompt: str, max_tokens: int = 128):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {
            "include_usage": True
        }
    }

    prompt_tokens = None
    completion_tokens = None
    content_chunks = []

    first_token_time = None
    start_time = time.perf_counter()

    async with client.stream("POST", f"{VLLM_BASE_URL}/v1/chat/completions", json=payload, headers=HEADERS) as response:
        response.raise_for_status()

        async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue

                data = line.removeprefix("data:").strip()

                if data == "[DONE]":
                    break

                event = json.loads(data)
                usage = event.get("usage")

                if usage is not None:
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")

                if not event.get("choices"):
                    continue

                content = event["choices"][0]["delta"].get("content")

                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    # print(content, end="", flush=True)
                    content_chunks.append(content)

        end_time = time.perf_counter()

        generated_text = "".join(content_chunks)
        # print("Generated text:", repr(generated_text))

        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time is not None else None
        e2e_ms = (end_time - start_time) * 1000

        return {
            "status_code": response.status_code,
            "generated_text": generated_text,
            "ttft_ms": ttft_ms,
            "e2e_ms": e2e_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }

async def run_batch(client: httpx.AsyncClient, prompts: list[str], max_tokens: int = 128):
        batch_start = time.perf_counter()
        results = await asyncio.gather(
            *(
                send_streaming_request(client, prompt, max_tokens)
                for prompt in prompts
            )
        )
        batch_end = time.perf_counter()
        batch_time_s = batch_end - batch_start

        return {
            "results": results,
            "batch_time_s": batch_time_s
        }

async def main():
    async with httpx.AsyncClient(timeout=600, follow_redirects=True) as client:
        test_prompt = "Explain what a KV cache is in one sentence."

        # warmup request
        print("=== Warmup Request ===")
        result = await send_streaming_request(client, test_prompt)

        print("Status: ", result["status_code"])
        print("Generated text:", repr(result["generated_text"]))
        if result["ttft_ms"] is not None:
            print(f"TTFT: {result['ttft_ms']:.2f} ms")
        else:
            print("TTFT: N/A (no generated text)")
        print(f"E2E: {result['e2e_ms']:.2f} ms")
        print(f"Prompt tokens: {result['prompt_tokens']}")
        print(f"Completion tokens: {result['completion_tokens']}")

        # batch requests
        print("\n=== Concurrent Batch ===")
        workloads = {
            "short": 256,
            "long": 2048
        }
        concurrency_levels = [1, 4, 8]
        num_trials = 5
        all_batches = []

        for workload_name, target_tokens in workloads.items():
            print(f"\n=== Workload: {workload_name} ({target_tokens} tokens) ===")

            for concurrency in concurrency_levels:
                print(f"\n--- Concurrency: {concurrency} ---")

                for trial in range(num_trials):
                    print(f"\nTrial {trial + 1}/{num_trials}")

                    prompts = []

                    for request_id in range(concurrency):
                        seed = (
                            target_tokens * 1_000_000
                            + concurrency * 10_000
                            + trial * 100
                            + request_id
                        )

                        prompt = generate_prompt(target_tokens, seed)
                        prompts.append(prompt)
                        
                    batch = await run_batch(client, prompts)

                    batch["workload"] = workload_name
                    batch["target_tokens"] = target_tokens
                    batch["concurrency"] = concurrency
                    batch["trial"] = trial + 1

                    results = batch["results"]
                    batch_time_s = batch["batch_time_s"]

                    output_tokens_per_sec = None

                    for i, result in enumerate(results):
                        ttft = result["ttft_ms"]
                        e2e = result["e2e_ms"]

                        ttft_display = f"{ttft:.2f}" if ttft is not None else "N/A"

                        print(
                            f"Request {i + 1}: "
                            f"TTFT={ttft_display} ms, "
                            f"E2E={e2e:.2f} ms"
                        )

                    if all(r["completion_tokens"] is not None for r in results):
                        total_completion_tokens = sum(r["completion_tokens"] for r in results)
                        output_tokens_per_sec = total_completion_tokens / batch_time_s
                        print(f"\nOutput tokens per second: {output_tokens_per_sec:.2f} tok/s")
                    else:
                        print("\nOutput tokens per second: N/A (missing token counts)")
                    batch["output_tokens_per_sec"] = output_tokens_per_sec

                    requests_per_sec = concurrency / batch_time_s
                    batch["requests_per_sec"] = requests_per_sec

                    print(f"Requests per second: {requests_per_sec:.2f} req/s")

                    print(f"Batch time: {(batch_time_s * 1000):.2f} ms")

                    all_batches.append(batch)

            print("Total batches collected so far:", len(all_batches))
            
            print("\n=== Latency Summary ===")
            for concurrency in concurrency_levels:

                selected_batches = [
                    batch for batch in all_batches 
                    if batch["workload"] == workload_name
                    and batch["concurrency"] == concurrency
                ]

                ttft_values = [
                    request["ttft_ms"]
                    for batch in selected_batches
                    for request in batch["results"]
                    if request["ttft_ms"] is not None
                ]

                e2e_values = [
                    request["e2e_ms"]
                    for batch in selected_batches
                    for request in batch["results"]
                ]

                token_tp_values = [
                    batch["output_tokens_per_sec"]
                    for batch in selected_batches
                    if batch["output_tokens_per_sec"] is not None
                ]

                request_tp_values = [
                    batch["requests_per_sec"]
                    for batch in selected_batches
                ]

                print(f"\n--- Concurrency: {concurrency} ---")
                print(f"TTFT samples: {len(ttft_values)}")
                print(f"E2E samples: {len(e2e_values)}")

                if len(ttft_values) >= 2:
                    ttft_p50 = statistics.median(ttft_values)
                    ttft_p95 = statistics.quantiles(ttft_values, n=100, method="inclusive")[94]
                    print(f"TTFT p50: {ttft_p50:.2f} ms")
                    print(f"TTFT p95: {ttft_p95:.2f} ms")
                else:
                    print("TTFT p50/p95: N/A (insufficient samples)")

                if len(e2e_values) >= 2:
                    e2e_p50 = statistics.median(e2e_values)
                    e2e_p95 = statistics.quantiles(e2e_values, n=100, method="inclusive")[94]
                    print(f"E2E p50: {e2e_p50:.2f} ms")
                    print(f"E2E p95: {e2e_p95:.2f} ms")
                else:
                    print("E2E p50/p95: N/A (insufficient samples)")

                if len(token_tp_values) >= 2:
                    token_tp_mean = statistics.mean(token_tp_values)
                    token_tp_stdev = statistics.stdev(token_tp_values)
                    print(f"Output tokens per second (mean): {token_tp_mean:.2f} tok/s")
                    print(f"Output tokens per second (stdev): {token_tp_stdev:.2f} tok/s")
                else:
                    print("Output tokens per second (mean/stdev): N/A (insufficient samples)")

                if len(request_tp_values) >= 2:
                    request_tp_mean = statistics.mean(request_tp_values)
                    request_tp_stdev = statistics.stdev(request_tp_values)
                    print(f"Requests per second (mean): {request_tp_mean:.2f} req/s")
                    print(f"Requests per second (stdev): {request_tp_stdev:.2f} req/s")
                else:
                    print("Requests per second (mean/stdev): N/A (insufficient samples)")

if __name__ == "__main__":
    asyncio.run(main())
