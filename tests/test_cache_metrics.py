import unittest
from pathlib import Path

from bench.cache_metrics import cache_configuration, interval_summary, is_idle, parse_metrics, process_identity
from server_config import MODEL_NAME

ROOT = Path(__file__).resolve().parents[1]

def metric_text(*, tokens=0, hits=0, prefill=0, preemptions=0, identity=100, expected=True):
    labels = f'model_name="{MODEL_NAME}",engine="0"'

    return "\n".join([
        f"process_start_time_seconds {identity}",
        f'vllm:cache_config_info{{enable_prefix_caching="{expected}",gpu_memory_utilization="0.8",engine="0"}} 1',
        f"vllm:num_requests_running{{{labels}}} 0",
        f"vllm:num_requests_waiting{{{labels}}} 0",
        f"vllm:prefix_cache_queries_total{{{labels}}} {tokens}",
        f"vllm:prefix_cache_hits_total{{{labels}}} {hits}",
        f"vllm:request_prefill_kv_computed_tokens_sum{{{labels}}} {prefill}",
        f"vllm:num_preemptions_total{{{labels}}} {preemptions}"
    ]) + "\n"

class CacheMetricsTests(unittest.TestCase):
    def test_historical_eight_request_diagnostic(self):
        before = parse_metrics((ROOT / "metrics" / "metrics-before-long.txt").read_text(encoding="utf-8"))
        after = parse_metrics((ROOT / "metrics" / "metrics-after-long.txt").read_text(encoding="utf-8"))
        summary = interval_summary(before, after)

        self.assertEqual(summary, {
            "prefix_queries_tokens": 16384,
            "prefix_hits_tokens": 0,
            "new_prefill_tokens": 16384,
            "preemptions": 0,
            "token_cache_hit_rate": 0
        })

    def test_hit_rate_and_zero_denominator(self):
        initial = parse_metrics(metric_text(tokens=20, hits=10, prefill=10))
        final = parse_metrics(metric_text(tokens=120, hits=90, prefill=30))
        summary = interval_summary(initial, final)

        self.assertEqual(summary["token_cache_hit_rate"], 0.8)
        self.assertEqual(summary["new_prefill_tokens"], 20)
        self.assertIsNone(interval_summary(initial, initial)["token_cache_hit_rate"])

    def test_missing_counters_remain_unavailable(self):
        text = "\n".join(line for line in metric_text().splitlines() if "prefix_cache_hits" not in line)
        summary = interval_summary(parse_metrics(text), parse_metrics(text))

        self.assertIsNone(summary["prefix_hits_tokens"])
        self.assertIsNone(summary["token_cache_hit_rate"])

    def test_restarts_counter_resets_and_label_changes_are_rejected(self):
        before_text = metric_text(tokens=100, hits=50)
        before = parse_metrics(before_text)

        for after_text in (
            metric_text(tokens=200, hits=80, identity=101),
            metric_text(tokens=90, hits=50),
            before_text.replace('prefix_cache_hits_total{', 'prefix_cache_hits_total{extra="changed",'),
            before_text + 'vllm:prefix_cache_hits_created{engine="0"} 101\n'
        ):
            with self.subTest(after=after_text), self.assertRaises(ValueError):
                interval_summary(before, parse_metrics(after_text))

    def test_running_treatment_and_idle_state_are_verified(self):
        samples = parse_metrics(metric_text(expected=False))

        self.assertEqual(process_identity(samples), 100)
        self.assertEqual(cache_configuration(samples, False)["enable_prefix_caching"], "False")
        self.assertTrue(is_idle(samples))

        with self.assertRaisesRegex(ValueError, "treatment"):
            cache_configuration(samples, True)

        for name in ("process_start_time_seconds", "vllm:num_requests_waiting", "vllm:cache_config_info"):
            incomplete = {key: value for key, value in samples.items() if key[0] != name}

            with self.subTest(name=name), self.assertRaises(ValueError):
                process_identity(incomplete)
                cache_configuration(incomplete, False)
                is_idle(incomplete)

    def test_invalid_values_and_impossible_hits_are_rejected(self):
        for text in (metric_text(tokens=-1), metric_text(tokens=float("nan")), metric_text(tokens=10, hits=11)):
            with self.subTest(text=text), self.assertRaises(ValueError):
                interval_summary(parse_metrics(metric_text()), parse_metrics(text))

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_metrics(metric_text() + "process_start_time_seconds 100\n")

if __name__ == "__main__":
    unittest.main()
