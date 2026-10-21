import subprocess
import sys
import unittest
from pathlib import Path

from bench import batch_results, compare, compare_batches, results

ROOT = Path(__file__).resolve().parents[1]

class CompatibilityTests(unittest.TestCase):
    def test_old_result_imports_use_the_batch_writers(self):
        self.assertIs(results.save_results_json, batch_results.save_results_json)
        self.assertIs(results.save_results_csv, batch_results.save_results_csv)

    def test_historical_datasets_remain_readable(self):
        for filename, expected in (
            ("20261019_014825.json", 30),
            ("20261019_024946.json", 40),
            ("20261019_030158.json", 40)
        ):
            with self.subTest(filename=filename):
                batches = compare_batches.load_batches(ROOT / "results" / filename)

                self.assertEqual(len(batches), expected)
                self.assertIs(compare.load_batches, compare_batches.load_batches)

    def test_historical_comparison_command_matches_the_renamed_command(self):
        arguments = [
            str(ROOT / "results" / "20261019_024946.json"),
            str(ROOT / "results" / "20261019_030158.json")
        ]
        outputs = []

        for module in ("bench.compare", "bench.compare_batches"):
            completed = subprocess.run(
                [sys.executable, "-m", module, *arguments],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True
            )
            outputs.append(completed.stdout)

        self.assertEqual(outputs[0], outputs[1])
        self.assertIn("+14.04%", outputs[0])
        self.assertIn("+1.51%", outputs[0])

if __name__ == "__main__":
    unittest.main()
