import argparse
from pathlib import Path

from bench.compare_cache import load_run, timestamp

def load_pilots(directory: Path) -> list:
    pilots = [
        load_run(directory / name, expected)
        for name, expected in (("pilot-off", False), ("pilot-on", True))
    ]
    (off, off_manifest, _), (on, on_manifest, _) = pilots

    if off_manifest["sha256"] != on_manifest["sha256"]:
        raise ValueError("Pilots must replay the same workload")
    if any(
        off["server"][field] != on["server"][field]
        for field in ("gpu_declared", "vllm_version_declared")
    ):
        raise ValueError("Pilots must use the same declared GPU and vLLM version")
    if (
        off["server"]["process_start_time_seconds"]
        == on["server"]["process_start_time_seconds"]
    ):
        raise ValueError("Pilots require different server processes")
    if timestamp(off["ended_at"]) > timestamp(on["started_at"]):
        raise ValueError("Pilots must run in cache-off then cache-on order")

    return pilots

def main():
    parser = argparse.ArgumentParser(
        description="Validate saved cache-off/on pilots and show their cache metrics."
    )
    parser.add_argument(
        "directory", type=Path, help="Study directory containing pilot-off and pilot-on"
    )
    args = parser.parse_args()

    try:
        pilots = load_pilots(args.directory)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))

    for name, (_, _, rows) in zip(("pilot-off", "pilot-on"), pilots):
        print(f"\n{name}: validated")
        for row in rows:
            print(
                f"  {row['scenario']}: "
                f"queries={row['prefix_queries_tokens']}, "
                f"hits={row['prefix_hits_tokens']}, "
                f"hit_rate={row['token_cache_hit_rate']}"
            )

if __name__ == "__main__":
    main()
