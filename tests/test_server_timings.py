import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bench.cache_metrics import counter_changes, parse_metrics
from bench.server_timings import TIMING_METRICS, timing_summary
from server_config import MODEL_NAME

QUEUE = TIMING_METRICS["queue"]


def histogram_text(
    *, seconds=0, count=0, created=10, metric=QUEUE, engine="0", model=MODEL_NAME
):
    labels = f'model_name="{model}",engine="{engine}"'
    return (
        f"{metric}_sum{{{labels}}} {seconds}\n"
        f"{metric}_count{{{labels}}} {count}\n"
        f"{metric}_created{{{labels}}} {created}\n"
    )


def snapshot(text="", *, identity=100, expected=True):
    return parse_metrics(
        f"process_start_time_seconds {identity}\n"
        f'vllm:cache_config_info{{enable_prefix_caching="{expected}"}} 1\n' + text
    )


class ServerTimingsTests(unittest.TestCase):
    def test_window_deltas_exclude_baseline_and_convert_seconds_to_ms(self):
        before = snapshot(
            "".join(
                histogram_text(seconds=20, count=8, metric=metric)
                for metric in TIMING_METRICS.values()
            )
        )
        after = snapshot(
            "".join(
                histogram_text(seconds=21, count=12, metric=metric)
                for metric in TIMING_METRICS.values()
            )
        )
        result = timing_summary(before, after)

        for field in TIMING_METRICS:
            with self.subTest(field=field):
                self.assertEqual(
                    result[field],
                    {
                        "observations": 4,
                        "total_seconds": 1,
                        "mean_ms": 250,
                    },
                )

    def test_engine_means_are_weighted_by_observation_counts(self):
        before = snapshot(histogram_text() + histogram_text(engine="1"))
        after = snapshot(
            histogram_text(seconds=0.1, count=1)
            + histogram_text(seconds=2.7, count=9, engine="1")
        )
        queue = timing_summary(before, after)["queue"]
        self.assertEqual(queue["observations"], 10)
        self.assertAlmostEqual(queue["mean_ms"], 280)

    def test_missing_histogram_components_are_unavailable(self):
        text = histogram_text()
        for absent in ("_sum", "_count", "both"):
            modified = "\n".join(
                line
                for line in text.splitlines()
                if absent != "both" and not line.startswith(QUEUE + absent)
            )
            with self.subTest(absent=absent):
                queue = timing_summary(snapshot(text), snapshot(modified))["queue"]
                self.assertIsNone(queue["observations"])
                self.assertIsNone(queue["total_seconds"])
                self.assertIsNone(queue["mean_ms"])

    def test_zero_observations_and_zero_duration_remain_distinct(self):
        before = snapshot(histogram_text())
        empty = timing_summary(before, before)["queue"]
        self.assertEqual(empty["observations"], 0)
        self.assertEqual(empty["total_seconds"], 0)
        self.assertIsNone(empty["mean_ms"])
        zero = timing_summary(before, snapshot(histogram_text(count=2)))["queue"]
        self.assertEqual(zero["observations"], 2)
        self.assertEqual(zero["mean_ms"], 0)

    def test_resets_restarts_and_creation_changes_are_rejected(self):
        before = snapshot(histogram_text(seconds=10, count=8))
        for after in (
            snapshot(histogram_text(seconds=9, count=10)),
            snapshot(histogram_text(seconds=11, count=7)),
            snapshot(histogram_text(seconds=11, count=10, created=11)),
            snapshot(histogram_text(seconds=11, count=10), identity=101),
            snapshot(histogram_text(seconds=11, count=10), expected=False),
        ):
            with self.subTest(after=after), self.assertRaises(ValueError):
                timing_summary(before, after)

    def test_histogram_count_checks_the_shared_creation_timestamp(self):
        before = snapshot(histogram_text(created=10))
        after = snapshot(histogram_text(created=11))
        with self.assertRaisesRegex(ValueError, "creation time"):
            counter_changes(before, after, QUEUE + "_count")

    def test_fractional_negative_and_nonfinite_values_are_rejected(self):
        for seconds, count in (
            (-1, 2),
            (float("nan"), 2),
            (float("inf"), 2),
            (1, -1),
            (1, 0.5),
            (1, float("nan")),
        ):
            with (
                self.subTest(seconds=seconds, count=count),
                self.assertRaises(ValueError),
            ):
                timing_summary(
                    snapshot(histogram_text()),
                    snapshot(histogram_text(seconds=seconds, count=count)),
                )
        with self.assertRaisesRegex(ValueError, "integers"):
            timing_summary(
                snapshot(histogram_text(count=0.5)),
                snapshot(histogram_text(seconds=1, count=1.5)),
            )

    def test_sum_without_observations_is_rejected_per_engine(self):
        before = snapshot(histogram_text() + histogram_text(engine="1"))
        after = snapshot(
            histogram_text(seconds=1, count=0)
            + histogram_text(seconds=1, count=1, engine="1")
        )
        with self.assertRaisesRegex(ValueError, "without observations"):
            timing_summary(before, after)

    def test_incompatible_labels_and_wrong_models_are_rejected(self):
        before_text = histogram_text()
        after_text = histogram_text(seconds=1, count=2)
        for changed in (
            after_text.replace('engine="0"', 'engine="1"'),
            after_text.replace(MODEL_NAME, "other-model"),
            after_text.replace(QUEUE + "_count{", QUEUE + '_count{extra="changed",'),
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                timing_summary(snapshot(before_text), snapshot(changed))

        mismatched = before_text.replace(
            QUEUE + "_count{", QUEUE + '_count{extra="changed",'
        )
        with self.assertRaisesRegex(ValueError, "sum/count labels"):
            timing_summary(snapshot(mismatched), snapshot(mismatched))

    def test_cli_reads_snapshots_and_prints_json(self):
        with TemporaryDirectory() as directory:
            before = Path(directory) / "before.prom"
            after = Path(directory) / "after.prom"
            before.write_text(
                "process_start_time_seconds 100\n" + histogram_text(), encoding="utf-8"
            )
            after.write_text(
                "process_start_time_seconds 100\n" + histogram_text(seconds=1, count=4),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bench.server_timings",
                    str(before),
                    str(after),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(json.loads(result.stdout)["queue"]["mean_ms"], 250)


if __name__ == "__main__":
    unittest.main()
