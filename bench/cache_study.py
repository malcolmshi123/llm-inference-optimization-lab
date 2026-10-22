import argparse
import asyncio
import csv
import io
import json
import math
import os
import platform
import shlex
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import httpx

from bench.async_client import positive_int, run_batch
from bench.cache_metrics import (
    cache_configuration,
    interval_summary,
    is_idle,
    parse_metrics,
    process_identity,
    validate_completed_requests,
)
from bench.cache_workloads import create_manifest, load_manifest
from bench.workloads import generate_prompt
from server_config import GPU_TYPE, VLLM_VERSION, ServerConfig

ROOT = Path(__file__).resolve().parents[1]

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def write_text(path: Path, text: str):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * 2**attempt)

def write_json(path: Path, value: dict):
    write_text(
        path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )

def save_run(directory: Path, run: dict):
    write_json(directory / "run.json", run)
    output = io.StringIO(newline="")
    fields = [
        "scenario",
        "concurrency",
        "request_id",
        "batch",
        "batch_time_s",
        "success",
        "ttft_ms",
        "e2e_ms",
        "prompt_tokens",
        "completion_tokens",
        "finish_reason",
        "error_type",
        "error_message",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()

    for scenario in run["scenarios"]:
        for index, batch in enumerate(scenario["batches"], start=1):
            for identifier, request in zip(batch["request_ids"], batch["results"]):
                writer.writerow(
                    {
                        **request,
                        "scenario": scenario["name"],
                        "concurrency": scenario["concurrency"],
                        "request_id": identifier,
                        "batch": index,
                        "batch_time_s": batch["batch_time_s"],
                    }
                )

    write_text(directory / "requests.csv", output.getvalue())

def startup_command(log: str, expected: bool) -> list[str]:
    marker = "Starting vLLM server with command: "
    lines = [
        line.partition(marker)[2].strip() for line in log.splitlines() if marker in line
    ]

    if not lines:
        raise ValueError("Server log must contain the printed vLLM startup command")

    command = shlex.split(lines[-1])

    if command != ServerConfig(prefix_caching=expected).command():
        raise ValueError(
            "Startup command does not match the fixed cache-study controls"
        )

    return command

def client_metadata() -> dict:
    def git(*arguments):
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", *arguments],
            cwd=ROOT,
            text=True,
        ).strip()

    return {
        "git_revision": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("httpx", "transformers", "jinja2", "prometheus-client")
        },
    }

async def run_study(
    client,
    manifest: dict,
    *,
    directory: Path,
    expected: bool,
    server_log: Path,
    previous_run: Path | None = None,
):
    log = server_log.read_text(encoding="utf-8")
    command = startup_command(log, expected)
    previous = (
        json.loads(previous_run.read_text(encoding="utf-8")) if previous_run else None
    )

    if previous is not None:
        identity = previous.get("server", {}).get("process_start_time_seconds")

        if (
            previous.get("kind") != "cache_study"
            or type(identity) not in (int, float)
            or not math.isfinite(identity)
            or identity <= 0
        ):
            raise ValueError("Previous run must contain a cache-study server identity")

    metadata = client_metadata()
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "workload.json", manifest)
    write_text(directory / "server.log", log)
    warmup = [
        [
            {
                "role": "user",
                "content": generate_prompt(
                    2048, 10**12 + manifest["seed"] * 100 + index
                ),
            }
        ]
        for index in range(8)
    ]
    run = {
        "schema_version": 1,
        "kind": "cache_study",
        "status": "running",
        "started_at": utc_now(),
        "manifest_sha256": manifest["sha256"],
        "client": metadata,
        "server": {
            "expected_prefix_caching": expected,
            "startup_command_from_log": command,
            "vllm_version_declared": VLLM_VERSION,
            "gpu_declared": GPU_TYPE,
            "process_start_time_seconds": None,
            "previous_process_start_time_seconds": previous["server"][
                "process_start_time_seconds"
            ]
            if previous
            else None,
        },
        "warmup": {
            "messages": warmup,
            "max_tokens": 128,
        },
        "snapshots": {},
        "scenarios": [],
    }
    save_run(directory, run)

    async def snapshot(name):
        response = await client.get("/metrics")
        response.raise_for_status()
        path = f"{name}.prom"
        write_text(directory / path, response.text)
        run["snapshots"][name] = {
            "path": path,
            "captured_at": utc_now(),
        }
        samples = parse_metrics(response.text)
        config = cache_configuration(samples, expected)
        identity = process_identity(samples)
        recorded = run["server"]["process_start_time_seconds"]

        if recorded is not None and recorded != identity:
            raise ValueError("Server restarted during the treatment session")

        if previous and identity == previous["server"]["process_start_time_seconds"]:
            raise ValueError(
                "A fresh server is required after the previous treatment session"
            )

        run["server"].update(
            process_start_time_seconds=identity, observed_cache_config=config
        )

        return samples

    async def drained_snapshot(name):
        deadline = time.perf_counter() + 30

        while True:
            samples = await snapshot(name)

            if is_idle(samples):
                return samples

            if time.perf_counter() >= deadline:
                raise TimeoutError("Requests did not drain within 30 seconds")

            await asyncio.sleep(1)

    try:
        await drained_snapshot("session-start")
        run["warmup"]["started_at"] = utc_now()
        run["warmup"].update(await run_batch(client, warmup))
        run["warmup"]["ended_at"] = utc_now()
        save_run(directory, run)

        if any(not request["success"] for request in run["warmup"]["results"]):
            raise ValueError("Engine warmup failed")

        await drained_snapshot("warmup-after")

        for source in manifest["scenarios"]:
            scenario = {
                "name": source["name"],
                "case": source["case"],
                "concurrency": source["concurrency"],
                "status": "preparing",
                "priming": None,
                "batches": [],
            }
            run["scenarios"].append(scenario)
            save_run(directory, run)

            if source["primer"]:
                priming_started = utc_now()
                scenario["priming"] = await run_batch(
                    client, [source["primer"]["messages"]], max_tokens=1
                )
                scenario["priming"].update(
                    started_at=priming_started, ended_at=utc_now()
                )
                save_run(directory, run)

                primer_result = scenario["priming"]["results"][0]

                if (
                    not primer_result["success"]
                    or primer_result["prompt_tokens"]
                    != source["primer"]["prompt_tokens"]
                ):
                    raise ValueError(
                        "Prefix priming failed or tokenizer counts differ from the server"
                    )

            before_name, after_name = (
                source["name"] + "-before",
                source["name"] + "-after",
            )
            before = await drained_snapshot(before_name)
            scenario.update(
                status="measuring", started_at=utc_now(), before_snapshot=before_name
            )
            save_run(directory, run)
            started = time.perf_counter()

            try:
                for offset in range(0, len(source["requests"]), source["concurrency"]):
                    selected = source["requests"][
                        offset : offset + source["concurrency"]
                    ]
                    batch = await run_batch(
                        client, [request["messages"] for request in selected]
                    )
                    batch["request_ids"] = [
                        request["request_id"] for request in selected
                    ]

                    for request, result in zip(selected, batch["results"]):
                        if result["success"] and (
                            result["prompt_tokens"] != request["prompt_tokens"]
                            or result["completion_tokens"] > request["max_tokens"]
                        ):
                            result.update(
                                success=False,
                                error_type="TokenCountMismatch",
                                error_message=(
                                    "Server token usage disagrees with the saved workload"
                                ),
                            )

                    scenario["batches"].append(batch)
                    save_run(directory, run)

                    if any(not result["success"] for result in batch["results"]):
                        break
            finally:
                scenario.update(
                    elapsed_s=time.perf_counter() - started, ended_at=utc_now()
                )

            after = await drained_snapshot(after_name)
            scenario.update(
                after_snapshot=after_name, metrics=interval_summary(before, after)
            )

            if any(
                not request["success"]
                for batch in scenario["batches"]
                for request in batch["results"]
            ):
                scenario["status"] = "failed"
                raise ValueError("Measured requests failed; retained partial results")

            validate_completed_requests(
                before,
                after,
                [
                    request
                    for batch in scenario["batches"]
                    for request in batch["results"]
                ],
            )
            scenario["status"] = "completed"
            save_run(directory, run)

        run["status"] = "completed"
    except BaseException as exc:
        run.update(
            status="interrupted"
            if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
            else "failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

        if run["scenarios"] and run["scenarios"][-1]["status"] != "completed":
            run["scenarios"][-1]["status"] = run["status"]

        raise
    finally:
        run["ended_at"] = utc_now()
        save_run(directory, run)

    return run

def main():
    parser = argparse.ArgumentParser(
        description="Prepare and run a focused prefix-cache study with manual deployments."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare",
        help="Save deterministic document requests without contacting the server",
    )
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--count", type=positive_int, default=32)
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser(
        "run", help="Run one treatment session against an already deployed server"
    )
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--expected-prefix-caching", choices=("off", "on"), required=True)
    run.add_argument("--server-log", type=Path, required=True)
    run.add_argument(
        "--previous-run",
        type=Path,
        help="Previous session's run.json, for checking a fresh server identity",
    )
    run.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.command == "prepare":
            if args.output.exists():
                raise ValueError("Workload output already exists")

            from transformers import AutoTokenizer
            from server_config import MODEL_NAME, MODEL_REVISION

            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, revision=MODEL_REVISION
            )
            manifest = create_manifest(tokenizer, seed=args.seed, count=args.count)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, manifest)
            print(f"Saved {4 * args.count} measured requests to {args.output}")
        else:
            manifest = load_manifest(args.manifest)
            missing = [
                name
                for name in ("VLLM_BASE_URL", "MODAL_PROXY_KEY", "MODAL_PROXY_SECRET")
                if not os.environ.get(name)
            ]

            if missing:
                raise ValueError("Missing environment variables: " + ", ".join(missing))

            async def execute():
                async with httpx.AsyncClient(
                    base_url=os.environ["VLLM_BASE_URL"].rstrip("/"),
                    headers={
                        "Modal-Key": os.environ["MODAL_PROXY_KEY"],
                        "Modal-Secret": os.environ["MODAL_PROXY_SECRET"],
                    },
                    timeout=600,
                    follow_redirects=False,
                ) as client:
                    await run_study(
                        client,
                        manifest,
                        directory=args.output_dir,
                        expected=args.expected_prefix_caching == "on",
                        server_log=args.server_log,
                        previous_run=args.previous_run,
                    )

            asyncio.run(execute())
            print(f"Saved treatment results to {args.output_dir}")
    except (
        OSError,
        ValueError,
        TimeoutError,
        httpx.HTTPError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.exit(1, f"Error: {exc}\n")

if __name__ == "__main__":
    main()
