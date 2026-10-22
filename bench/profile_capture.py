import argparse
import asyncio
import os
import subprocess
import time
from pathlib import Path

import httpx

from bench.async_client import run_batch
from bench.cache_metrics import (
    cache_configuration,
    interval_summary,
    is_idle,
    parse_metrics,
    process_identity,
    validate_completed_requests,
)
from bench.cache_study import client_metadata, utc_now, write_json, write_text
from bench.cache_workloads import load_manifest
from bench.workloads import generate_prompt
from server_config import PROFILE_VOLUME_NAME, ServerConfig

async def capture(
    client,
    manifest: dict,
    *,
    directory: Path,
    expected: bool,
    profile_session: str,
    server_log: Path,
):
    config = ServerConfig(prefix_caching=expected, profile_session=profile_session)
    log = server_log.read_text(encoding="utf-8")
    commands = [
        line.partition("Starting vLLM server with command: ")[2].strip()
        for line in log.splitlines()
        if "Starting vLLM server with command: " in line
    ]

    if not commands or commands[-1] != " ".join(config.command()):
        raise ValueError("Startup log does not match the profiling configuration")

    source = next(s for s in manifest["scenarios"] if s["name"] == "shared-c8")
    selected = source["requests"][:8]

    if len(selected) != 8 or source["primer"] is None:
        raise ValueError(
            "Profiling requires at least eight shared-c8 requests and a primer"
        )

    run = {
        "schema_version": 1,
        "kind": "profile_capture",
        "status": "running",
        "started_at": utc_now(),
        "manifest_sha256": manifest["sha256"],
        "client": client_metadata(),
        "server": {
            "expected_prefix_caching": expected,
            "startup_command_from_log": config.command(),
            "process_start_time_seconds": None,
        },
        "scenario": "shared-c8",
        "request_ids": [request["request_id"] for request in selected],
        "profiler": {
            "session": profile_session,
            "volume": PROFILE_VOLUME_NAME,
            "directory": config.profile_directory,
            "start_attempted": False,
            "start_completed": False,
            "stop_attempted": False,
            "stop_completed": False,
            "trace_files_verified": False,
        },
        "snapshots": {},
    }
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "workload.json", manifest)
    write_text(directory / "server.log", log)

    def save():
        write_json(directory / "capture.json", run)

    async def snapshot(name):
        deadline = time.perf_counter() + 30

        while True:
            response = await client.get("/metrics")
            response.raise_for_status()
            write_text(directory / f"{name}.prom", response.text)
            samples = parse_metrics(response.text)
            cache_configuration(samples, expected)
            identity = process_identity(samples)
            recorded = run["server"]["process_start_time_seconds"]

            if recorded is not None and identity != recorded:
                raise ValueError("Server restarted during the profiling session")

            run["server"]["process_start_time_seconds"] = identity
            run["snapshots"][name] = {
                "path": f"{name}.prom",
                "captured_at": utc_now(),
            }

            if is_idle(samples):
                return samples

            if time.perf_counter() >= deadline:
                raise TimeoutError("Requests did not drain within 30 seconds")

            await asyncio.sleep(1)

    async def batch(name, messages, *, max_tokens=128, prompt_tokens=None):
        record = {
            "messages": messages,
            "max_tokens": max_tokens,
            "started_at": utc_now(),
        }
        run[name] = record

        try:
            record.update(await run_batch(client, messages, max_tokens=max_tokens))
        finally:
            record["ended_at"] = utc_now()
            if name != "captured_batch":
                save()

        for index, result in enumerate(record["results"]):
            if result["success"] and (
                result["completion_tokens"] > max_tokens
                or (
                    prompt_tokens is not None
                    and result["prompt_tokens"] != prompt_tokens[index]
                )
            ):
                result.update(
                    success=False,
                    error_type="TokenCountMismatch",
                    error_message="Server token usage disagrees with the saved workload",
                )

        if name != "captured_batch":
            save()

        if any(not result["success"] for result in record["results"]):
            raise ValueError(f"{name} requests failed; retained request results")

    save()
    failure = None

    try:
        await snapshot("session-start")
        await batch(
            "warmup",
            [
                [
                    {
                        "role": "user",
                        "content": generate_prompt(
                            2048, 10**12 + manifest["seed"] * 100 + index
                        ),
                    }
                ]
                for index in range(8)
            ],
        )
        await snapshot("warmup-after")
        await batch(
            "priming",
            [source["primer"]["messages"]],
            max_tokens=1,
            prompt_tokens=[source["primer"]["prompt_tokens"]],
        )
        before = await snapshot("capture-before")
        run["profiler"].update(start_attempted=True, start_requested_at=utc_now())
        save()
        response = await client.post("/start_profile")
        response.raise_for_status()
        run["profiler"].update(start_completed=True, started_at=utc_now())
        await batch(
            "captured_batch",
            [request["messages"] for request in selected],
            prompt_tokens=[request["prompt_tokens"] for request in selected],
        )
    except BaseException as exc:
        failure = exc
    finally:
        if run["profiler"]["start_attempted"]:
            run["profiler"].update(stop_attempted=True, stop_requested_at=utc_now())

            try:
                response = await client.post("/stop_profile")
                response.raise_for_status()
                run["profiler"].update(stop_completed=True, stopped_at=utc_now())
            except BaseException as exc:
                run["profiler"]["stop_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                if failure is None:
                    failure = exc

        if failure is None:
            try:
                after = await snapshot("capture-after")
                results = run["captured_batch"]["results"]
                validate_completed_requests(before, after, results)
                run["metrics"] = interval_summary(before, after)
            except BaseException as exc:
                failure = exc

        run.update(
            status="completed"
            if failure is None
            else (
                "interrupted"
                if isinstance(failure, (KeyboardInterrupt, asyncio.CancelledError))
                else "failed"
            ),
            ended_at=utc_now(),
        )
        if failure is not None:
            run.update(error_type=type(failure).__name__, error_message=str(failure))
        save()

    if failure is not None:
        raise failure

    return run

def main():
    parser = argparse.ArgumentParser(
        description="Profile one eight-request shared-prefix batch."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-prefix-caching", choices=("off", "on"), required=True
    )
    parser.add_argument("--profile-session", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
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
                timeout=300,
                follow_redirects=False,
            ) as client:
                await capture(
                    client,
                    manifest,
                    directory=args.output_dir,
                    expected=args.expected_prefix_caching == "on",
                    profile_session=args.profile_session,
                    server_log=args.server_log,
                )

        asyncio.run(execute())
    except (OSError, ValueError, httpx.HTTPError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print(f"Saved diagnostic capture record to {args.output_dir}")
    print(
        "Download and verify the trace files separately before stopping the deployment."
    )

if __name__ == "__main__":
    main()
