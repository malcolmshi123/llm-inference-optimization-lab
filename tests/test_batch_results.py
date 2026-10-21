from pathlib import Path
import json
import csv
import unittest
from bench.batch_results import save_results_json, save_results_csv
from tempfile import TemporaryDirectory

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
                "completion_tokens": 32,
            },
            {
                "success": False,
                "error_type": "ReadTimeout",
                "error_message": "Request timed out",
                "ttft_ms": None,
                "e2e_ms": None,
                "prompt_tokens": None,
                "completion_tokens": None,
            },
        ],
    }
]


class BatchResultsTests(unittest.TestCase):
    def test_json_preserves_successful_and_failed_requests(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "test.json"
            save_results_json(path, dummy_batches)

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")), dummy_batches
            )

    def test_csv_preserves_request_fields_and_empty_failure_values(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "test.csv"
            save_results_csv(path, dummy_batches)

            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["success"], "True")
        self.assertEqual(rows[1]["success"], "False")
        self.assertEqual(rows[1]["error_type"], "ReadTimeout")
        self.assertEqual(rows[1]["ttft_ms"], "")
        self.assertEqual(rows[1]["e2e_ms"], "")
        self.assertEqual(rows[0]["workload"], "short")
        self.assertEqual(rows[0]["concurrency"], "2")
        self.assertEqual(rows[0]["target_tokens"], "256")
        self.assertEqual(rows[0]["prompt_tokens"], "256")
        self.assertEqual(rows[0]["completion_tokens"], "32")
        self.assertEqual(float(rows[0]["batch_time_s"]), 1.2)


if __name__ == "__main__":
    unittest.main()
