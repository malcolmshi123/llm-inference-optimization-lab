import hashlib
import json
import random
from importlib.metadata import version
from pathlib import Path

from server_config import MAX_MODEL_LENGTH, MODEL_NAME, MODEL_REVISION

FIXTURES = Path(__file__).with_name("fixtures") / "documents.json"
SCENARIOS = (("unique", 1), ("shared", 1), ("unique", 8), ("shared", 8))
TARGET_TOKENS = 2048
OUTPUT_LIMIT = 128

def token_ids(tokenizer, messages: list[dict]) -> list[int]:
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True
    )["input_ids"]

def common_prefix(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index

    return min(len(left), len(right))

def fit_words(words: list[str], measure, target: int) -> str:
    low, high = 0, len(words)

    while low < high:
        middle = (low + high + 1) // 2

        if measure(" ".join(words[:middle])) <= target:
            low = middle
        else:
            high = middle - 1

    return " ".join(words[:low])

def manifest_digest(manifest: dict) -> str:
    payload = {key: value for key, value in manifest.items() if key != "sha256"}

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()

def create_manifest(tokenizer, *, seed: int = 42, count: int = 32) -> dict:
    if type(seed) is not int or type(count) is not int or count <= 0:
        raise ValueError("seed must be an integer and count must be a positive integer")

    if not tokenizer.chat_template:
        raise ValueError("The tokenizer must provide a chat template")

    documents = json.loads(FIXTURES.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    scenarios = []
    all_ids = []

    for case, concurrency in SCENARIOS:
        name = f"{case}-c{concurrency}"
        namespace = hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:16]
        document = documents[rng.randrange(len(documents))]
        words = " ".join(document["paragraphs"]).split() * 32

        def messages(text, question_id, question):
            return [
                {
                    "role": "user",
                    "content": (
                        f"{namespace}\nDocument:\n{text}\n\n"
                        f"Question {question_id}: {question}"
                    ),
                }
            ]

        prefix = ""
        primer = None

        if case == "shared":
            prefix = fit_words(
                words,
                lambda text: len(
                    token_ids(
                        tokenizer, messages(text, "primer", "Read this document.")
                    )
                ),
                int(TARGET_TOKENS * 0.9),
            )
            primer_messages = messages(prefix, "primer", "Read this document.")
            primer = {
                "messages": primer_messages,
                "max_tokens": 1,
                "prompt_tokens": len(token_ids(tokenizer, primer_messages)),
            }

        requests = []

        for index in range(count):
            identifier = hashlib.sha256(f"{namespace}:{index}".encode()).hexdigest()[
                :16
            ]
            question = document["questions"][index % len(document["questions"])]

            if case == "shared":

                def make_messages(details):
                    return messages(
                        prefix,
                        identifier,
                        question + "\nAdditional context:\n" + details,
                    )
            else:

                def make_messages(details):
                    return [
                        {
                            "role": "user",
                            "content": (
                                f"{identifier}\nDocument:\n{details}\n\n"
                                f"Question {identifier}: {question}"
                            ),
                        }
                    ]

            offset = rng.randrange(len(document["paragraphs"]))
            pool = (
                " ".join(
                    document["paragraphs"][offset:] + document["paragraphs"][:offset]
                ).split()
                * 32
            )
            details = fit_words(
                pool,
                lambda text: len(token_ids(tokenizer, make_messages(text))),
                TARGET_TOKENS,
            )
            request_messages = make_messages(details)
            ids = token_ids(tokenizer, request_messages)
            shared = (
                common_prefix(ids, token_ids(tokenizer, primer["messages"]))
                if primer
                else None
            )

            if (
                abs(len(ids) - TARGET_TOKENS) > TARGET_TOKENS * 0.05
                or len(ids) + OUTPUT_LIMIT > MAX_MODEL_LENGTH
            ):
                raise ValueError("Request length is outside the study limits")

            if shared is not None and abs(shared / len(ids) - 0.9) > 0.05:
                raise ValueError(
                    "Shared token prefix is outside 90% +/- 5 percentage points"
                )

            requests.append(
                {
                    "request_id": f"{name}-{identifier}",
                    "messages": request_messages,
                    "prompt_tokens": len(ids),
                    "max_tokens": OUTPUT_LIMIT,
                    "prefix_group": name if primer else None,
                    "shared_prefix_tokens": shared,
                }
            )
            all_ids.append((name, case, ids))

        scenarios.append(
            {
                "name": name,
                "case": case,
                "concurrency": concurrency,
                "primer": primer,
                "requests": requests,
            }
        )

    for index, (name, case, ids) in enumerate(all_ids):
        for other_name, _, other_ids in all_ids[:index]:
            if name != other_name or case == "unique":
                if common_prefix(ids, other_ids) > TARGET_TOKENS * 0.05:
                    raise ValueError(
                        "Unintended shared prefix between requests or scenarios"
                    )

    manifest = {
        "schema_version": 1,
        "kind": "cache_workload",
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "tokenizer_class": type(tokenizer).__name__,
        "transformers_version": version("transformers"),
        "chat_template_sha256": hashlib.sha256(
            json.dumps(tokenizer.chat_template, sort_keys=True).encode()
        ).hexdigest(),
        "fixture_sha256": hashlib.sha256(FIXTURES.read_bytes()).hexdigest(),
        "seed": seed,
        "count_per_scenario": count,
        "generation": {
            "temperature": 0,
            "ignore_eos": False,
        },
        "scenarios": scenarios,
    }
    manifest["sha256"] = manifest_digest(manifest)

    return manifest

def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))

    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != "cache_workload"
    ):
        raise ValueError("Expected a version 1 cache workload")

    if manifest.get("sha256") != manifest_digest(manifest):
        raise ValueError("Workload digest does not match its contents")

    if (
        manifest.get("model") != MODEL_NAME
        or manifest.get("revision") != MODEL_REVISION
    ):
        raise ValueError("Workload does not use the pinned model/tokenizer revision")

    if manifest.get("generation") != {
        "temperature": 0,
        "ignore_eos": False,
    }:
        raise ValueError("Unsupported generation settings")

    count = manifest.get("count_per_scenario")

    if type(count) is not int or count <= 0 or type(manifest.get("seed")) is not int:
        raise ValueError("Invalid request count or seed")

    scenarios = manifest.get("scenarios")

    if (
        not isinstance(scenarios, list)
        or any(not isinstance(s, dict) for s in scenarios)
        or [(s.get("case"), s.get("concurrency")) for s in scenarios] != list(SCENARIOS)
    ):
        raise ValueError("Expected the four ordered cache scenarios")

    identifiers = set()

    for scenario in scenarios:
        if (
            scenario.get("name") != f"{scenario['case']}-c{scenario['concurrency']}"
            or not isinstance(scenario.get("requests"), list)
            or len(scenario["requests"]) != count
        ):
            raise ValueError("Invalid scenario name or request count")

        primer = scenario.get("primer")

        if (scenario["case"] == "shared" and not isinstance(primer, dict)) or (
            scenario["case"] == "unique" and primer is not None
        ):
            raise ValueError(
                "Shared scenarios require a primer; unique scenarios do not"
            )

        for request in scenario["requests"] + ([primer] if primer else []):
            if not isinstance(request, dict):
                raise ValueError("Expected a request object")

            messages = request.get("messages")
            tokens = request.get("prompt_tokens")
            limit = 1 if request is primer else OUTPUT_LIMIT

            if (
                not isinstance(messages, list)
                or len(messages) != 1
                or not isinstance(messages[0], dict)
                or messages[0].get("role") != "user"
                or not isinstance(messages[0].get("content"), str)
            ):
                raise ValueError("Expected one user message per request")

            if (
                type(tokens) is not int
                or tokens <= 0
                or request.get("max_tokens") != limit
                or tokens + limit > MAX_MODEL_LENGTH
            ):
                raise ValueError("Invalid token counts or output limit")

            if request is primer:
                continue

            identifier = request.get("request_id")

            if (
                not isinstance(identifier, str)
                or not identifier
                or identifier in identifiers
            ):
                raise ValueError("Request IDs must be unique nonempty strings")

            identifiers.add(identifier)

            if abs(tokens - TARGET_TOKENS) > TARGET_TOKENS * 0.05:
                raise ValueError("Request is outside the input length tolerance")

            shared = request.get("shared_prefix_tokens")

            if primer and (
                type(shared) is not int
                or abs(shared / tokens - 0.9) > 0.05
                or request.get("prefix_group") != scenario["name"]
            ):
                raise ValueError("Invalid shared prefix metadata")

            if not primer and (
                shared is not None or request.get("prefix_group") is not None
            ):
                raise ValueError("Unique requests must not declare shared prefixes")

    return manifest
