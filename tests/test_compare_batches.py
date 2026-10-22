import subprocess
import sys
import unittest
from pathlib import Path

from bench import compare_batches

ROOT = Path(__file__).resolve().parents[1]

class CompareBatchesTests(unittest.TestCase):
    def test_historical_datasets_remain_readable(self):
        for filename, expected in (
            ("20261019_014825.json", 30),
            ("20261019_024946.json", 40),
            ("20261019_030158.json", 40),
        ):
            with self.subTest(filename=filename):
                batches = compare_batches.load_batches(ROOT / "results" / filename)

                self.assertEqual(len(batches), expected)

    def test_comparison_command_preserves_the_historical_negative_result(self):
        arguments = [
            str(ROOT / "results" / "20261019_024946.json"),
            str(ROOT / "results" / "20261019_030158.json"),
        ]
        completed = subprocess.run(
            [sys.executable, "-m", "bench.compare_batches", *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn("+14.04%", completed.stdout)
        self.assertIn("+1.51%", completed.stdout)

if __name__ == "__main__":
    unittest.main()
