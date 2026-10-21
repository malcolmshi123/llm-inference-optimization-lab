import csv
import json
from pathlib import Path

def save_results_json(path: str, batches: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    serialized_batches = json.dumps(batches, indent=2)
    Path(path).write_text(serialized_batches, encoding="utf-8")

def save_results_csv(path: str, batches: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for batch in batches:
        for request_id, request in enumerate(batch["results"]):
            row = {
                "workload": batch["workload"],
                "concurrency": batch["concurrency"],
                "trial": batch["trial"],
                "max_num_batched_tokens": batch.get("max_num_batched_tokens"),

                "request_id": request_id,
                "ttft_ms": request["ttft_ms"],
                "e2e_ms": request["e2e_ms"],

                "target_tokens": batch["target_tokens"],
                "prompt_tokens": request["prompt_tokens"],
                "completion_tokens": request["completion_tokens"],

                "batch_time_s": batch["batch_time_s"],
                "requests_per_sec": batch["requests_per_sec"],
                "output_tokens_per_sec": batch["output_tokens_per_sec"],

                "success": request["success"],
                "error_type": request.get("error_type"),
                "error_message": request.get("error_message")
            }
            rows.append(row)

    fieldnames = [
        "workload",
        "target_tokens",
        "concurrency",
        "max_num_batched_tokens",

        "trial",
        "request_id",
        "prompt_tokens",

        "completion_tokens",
        "ttft_ms",
        "e2e_ms",

        "batch_time_s",
        "requests_per_sec",
        "output_tokens_per_sec",

        "success",
        "error_type",
        "error_message"
    ]
    
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
