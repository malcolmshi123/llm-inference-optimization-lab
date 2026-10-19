from pathlib import Path
import json
import csv
from bench.results import save_results_json
from bench.results import save_results_csv

dummy_batches = [
    {
        "workload": "short",
        "target_tokens": 256,
        "concurrency": 2,
        "successful_requests": 1,
        "failed_requests": 1,
        "trial": 1,
        "batch_time_s": 1.2,
        "requests_per_sec": 0.8333,
        "output_tokens_per_sec": 26.6667,
        "results": [
            {
                "success": True,
                "ttft_ms": 250.0,
                "e2e_ms": 1200.0,
                "prompt_tokens": 256,
                "completion_tokens": 32
            },
            {
                "success": False,
                "error_type": "ReadTimeout",
                "error_message": "Request timed out",
                "ttft_ms": None,
                "e2e_ms": None,
                "prompt_tokens": None,
                "completion_tokens": None
            }
        ]
    }
]

# json
save_results_json("results/test.json", dummy_batches)

loaded = json.loads(Path("results/test.json").read_text(encoding="utf-8"))

assert loaded == dummy_batches
print("JSON persistence test passed!")

# csv
save_results_csv("results/test.csv", dummy_batches)

with open("results/test.csv", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

assert len(rows) == 2
assert rows[0]["success"] == "True"
assert rows[1]["success"] == "False"
assert rows[1]["error_type"] == "ReadTimeout"
assert rows[1]["ttft_ms"] == ""
assert rows[1]["e2e_ms"] == ""
assert rows[0]["workload"] == "short"
assert rows[0]["concurrency"] == "2"
assert rows[0]["target_tokens"] == "256"
assert rows[0]["prompt_tokens"] == "256"
assert rows[0]["completion_tokens"] == "32"
assert float(rows[0]["batch_time_s"]) == 1.2

print("CSV persistence test passed!")
