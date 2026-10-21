import copy
import hashlib
import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bench.cache_workloads import (
    common_prefix,
    create_manifest,
    load_manifest,
    manifest_digest,
    token_ids,
)
from server_config import MODEL_NAME, MODEL_REVISION


class WordTokenizer:
    chat_template = "word-tokenizer-template"

    def apply_chat_template(self, messages, **kwargs):
        words = re.findall(r"\w+|[^\w\s]", messages[0]["content"])
        ids = [int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) for word in words]

        return {
            "input_ids": [1, *ids, 2],
        }


class CacheWorkloadTests(unittest.TestCase):
    def test_seed_replays_identical_payloads_and_other_seeds_change_them(self):
        tokenizer = WordTokenizer()
        first = create_manifest(tokenizer, count=2)

        self.assertEqual(first, create_manifest(tokenizer, count=2))
        self.assertNotEqual(first, create_manifest(tokenizer, count=2, seed=43))

    def test_lengths_prefixes_unique_questions_and_scenario_isolation(self):
        tokenizer = WordTokenizer()
        manifest = create_manifest(tokenizer, count=8)
        seen_payloads = set()
        prefixes = []

        for scenario in manifest["scenarios"]:
            requests = scenario["requests"]

            for request in requests:
                ids = token_ids(tokenizer, request["messages"])

                self.assertLessEqual(abs(len(ids) - 2048), 2048 * 0.05)
                self.assertEqual(request["prompt_tokens"], len(ids))
                self.assertLessEqual(len(ids) + request["max_tokens"], 4096)
                self.assertEqual(request["max_tokens"], 128)
                payload = json.dumps(request["messages"])
                self.assertNotIn(payload, seen_payloads)
                seen_payloads.add(payload)

                if scenario["primer"]:
                    shared = common_prefix(
                        ids, token_ids(tokenizer, scenario["primer"]["messages"])
                    )

                    self.assertEqual(shared, request["shared_prefix_tokens"])
                    self.assertLessEqual(abs(shared / len(ids) - 0.9), 0.05)
                else:
                    self.assertLessEqual(
                        common_prefix(
                            ids, token_ids(tokenizer, requests[0]["messages"])
                        )
                        if request is not requests[0]
                        else 0,
                        102,
                    )

            prefixes.append(token_ids(tokenizer, requests[0]["messages"]))

        for index, ids in enumerate(prefixes):
            for other_ids in prefixes[:index]:
                self.assertLessEqual(common_prefix(ids, other_ids), 102)

    def test_manifest_round_trip_and_modified_payload_rejection(self):
        manifest = create_manifest(WordTokenizer(), count=1)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(load_manifest(path), manifest)
            manifest["scenarios"][0]["requests"][0]["messages"][0]["content"] += (
                " changed"
            )
            path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "digest"):
                load_manifest(path)

    def test_invalid_saved_controls_and_context_are_rejected(self):
        original = create_manifest(WordTokenizer(), count=1)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"

            for field, value in (
                ("prompt_tokens", 4096),
                ("max_tokens", 256),
                ("request_id", ""),
            ):
                manifest = copy.deepcopy(original)
                manifest["scenarios"][0]["requests"][0][field] = value
                manifest["sha256"] = manifest_digest(manifest)
                path.write_text(json.dumps(manifest), encoding="utf-8")

                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_manifest(path)

    def test_invalid_generator_arguments_are_rejected(self):
        for options in (
            {
                "count": 0,
            },
            {
                "count": True,
            },
            {
                "seed": False,
            },
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                create_manifest(WordTokenizer(), **options)

    def test_cached_model_tokenizer_confirms_real_lengths_and_prefixes(self):
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, revision=MODEL_REVISION, local_files_only=True
            )
        except OSError:
            self.skipTest("Pinned tokenizer is not cached locally")

        for seed in (42, 43):
            manifest = create_manifest(tokenizer, seed=seed, count=32)

            for scenario in manifest["scenarios"]:
                for request in scenario["requests"]:
                    ids = token_ids(tokenizer, request["messages"])

                    self.assertLessEqual(abs(len(ids) - 2048), 2048 * 0.05)

                    if scenario["primer"]:
                        self.assertLessEqual(
                            abs(
                                common_prefix(
                                    ids,
                                    token_ids(
                                        tokenizer, scenario["primer"]["messages"]
                                    ),
                                )
                                / len(ids)
                                - 0.9
                            ),
                            0.05,
                        )


if __name__ == "__main__":
    unittest.main()
