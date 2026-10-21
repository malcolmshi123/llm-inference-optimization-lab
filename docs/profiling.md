# Capture a small profiling pair

Run these commands in **Git Bash**, from the repository root. They use the existing A10 deployment and capture eight shared-prefix requests per treatment. GPU startup and image preparation may take longer than the capture itself. This procedure has been checked offline; the first live capture is still a validation step.

Use one terminal. Run cache-off first and continue to cache-on only after its trace download verifies. Do not run the normal benchmark helper or another client against the app during either capture.

## 1. Set up the shell

```bash
cd /c/Users/mshi/vscode-ws/llm-inference-optimization-lab
PY="$PWD/.venv/Scripts/python.exe"
APP=llm-inference-optimization-lab
PROFILE_DIR="$PWD/inference-runs/profiling"
MANIFEST="$PWD/inference-runs/cache-study-live-01/workloads/round-1.json"
VOLUME=inference-lab-profiles

: "${VLLM_BASE_URL:?Set the deployed endpoint root, without /v1}"
: "${MODAL_PROXY_KEY:?Set your Modal proxy key}"
: "${MODAL_PROXY_SECRET:?Set your Modal proxy secret}"
export VLLM_BASE_URL MODAL_PROXY_KEY MODAL_PROXY_SECRET
unset INFERENCE_LAB_PROFILE_SESSION

test -x "$PY"
test -f "$MANIFEST"
"$PY" -m bench.profile_capture --help
```

Keep credentials in the shell environment. The app name and endpoint must refer to the same deployment. If Modal prints a different endpoint during deployment, stop and update `VLLM_BASE_URL` before trying a capture.

## 2. Define the download check

This reads every downloaded compressed trace, requires at least one GPU kernel event, and writes a separate validation receipt. CPU-only frontend traces are allowed alongside GPU worker traces. It does not change `capture.json` or establish an optimization result.

```bash
check_profile_traces() {
  "$PY" - "$1" <<'PY'
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
run = json.loads((root / "capture.json").read_text(encoding="utf-8"))
assert run["kind"] == "profile_capture" and run["status"] == "completed"
assert run["profiler"]["stop_completed"]
files = sorted((root / "traces").rglob("*.json.gz"))
assert files, "No compressed trace files were downloaded"
records = []
for path in files:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        trace = json.load(handle)
    kernels = sum(
        event.get("cat") == "kernel" and event.get("dur", 0) > 0
        for event in trace["traceEvents"]
    )
    records.append({
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "gpu_kernel_events": kernels,
    })
assert sum(record["gpu_kernel_events"] for record in records) > 0, (
    "Traces contain no positive-duration GPU kernel events"
)
receipt = {
    "kind": "profile_trace_check",
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "traces": records,
}
(root / "trace-check.json").write_text(
    json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
)
print("Verified traces:", len(files))
print("GPU kernel events:", sum(record["gpu_kernel_events"] for record in records))
PY
}
```

## 3. Define the treatment procedure

The helper deploys exactly one profiling treatment, checks readiness and startup evidence, runs the capture driver, downloads traces, and stops the app. Its exit trap also attempts shutdown on failure or Ctrl+C. It refuses existing local session/log directories. Use a fresh label for every attempted treatment, including retries, so remote trace directories cannot mix attempts.

Readiness is limited to 300 seconds. Startup-log polling and download retries are bounded; a failed capture is not automatically rerun. Modal may need a few seconds to commit flushed files to its Volume, so a download can be retried in the same newly created trace directory. The `--force` below only refreshes that session's downloaded copies. [Volume persistence](https://modal.com/docs/guide/volumes#background-commits)

```bash
capture_profile() (
  set -euo pipefail
  : "${VLLM_BASE_URL:?Set the endpoint root}"
  : "${MODAL_PROXY_KEY:?Set your Modal proxy key}"
  : "${MODAL_PROXY_SECRET:?Set your Modal proxy secret}"
  test -x "$PY"
  test -f "$MANIFEST"
  test -f modal_app.py
  declare -F check_profile_traces >/dev/null
  if (( $# < 2 || $# > 3 )); then
    printf 'Usage: capture_profile NAME off|on [PREVIOUS_NAME]\n' >&2
    exit 1
  fi
  name="$1"
  expected="$2"
  previous="${3:-}"
  [[ "$name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]]
  case "$expected" in
    off) cache=false ;;
    on) cache=true ;;
    *) printf 'Treatment must be off or on\n' >&2; exit 1 ;;
  esac
  output="$PROFILE_DIR/$name"
  logs="$PROFILE_DIR/logs/$name"
  if [[ -e "$output" || -e "$logs" ]]; then
    printf 'Session already exists: %s\nUse a fresh label.\n' "$name" >&2
    exit 1
  fi
  previous_capture=""
  if [[ -n "$previous" ]]; then
    previous_capture="$PROFILE_DIR/$previous/capture.json"
    test -f "$previous_capture"
  fi
  mkdir -p "$logs"
  trap '"$PY" -m modal app stop "$APP" --env main --yes' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  since=$("$PY" -c \
    'from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat())')

  INFERENCE_LAB_PREFIX_CACHING="$cache" \
  INFERENCE_LAB_MAX_NUM_BATCHED_TOKENS=2048 \
  INFERENCE_LAB_PROFILE_SESSION="$name" \
  "$PY" -m modal deploy modal_app.py --env main

  "$PY" -m modal function autoscaler update "$APP/serve" \
    --env main --min-containers 1

  curl --location --fail --silent --show-error --max-time 300 \
    -H "Modal-Key: $MODAL_PROXY_KEY" \
    -H "Modal-Secret: $MODAL_PROXY_SECRET" \
    --output "$logs/ready.prom" \
    --write-out 'Readiness: HTTP %{http_code}, %{time_total}s, %{size_download} bytes\n' \
    "${VLLM_BASE_URL%/}/metrics"

  "$PY" - "$logs/ready.prom" "$expected" "$previous_capture" <<'PY'
import json
import sys
from pathlib import Path
from bench.cache_metrics import cache_configuration, is_idle, parse_metrics, process_identity

samples = parse_metrics(Path(sys.argv[1]).read_text(encoding="utf-8"))
cache_configuration(samples, sys.argv[2] == "on")
identity = process_identity(samples)
assert is_idle(samples), "Server is handling other requests"
if sys.argv[3]:
    previous = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
    assert previous["kind"] == "profile_capture" and previous["status"] == "completed"
    assert identity != previous["server"]["process_start_time_seconds"], "Server was reused"
print("Ready server identity:", identity)
PY

  found=false
  for attempt in {1..10}; do
    "$PY" -m modal app logs "$APP" --env main --since "$since" \
      --source stdout --tail 1000 > "$logs/server.log"
    if grep -qF "Profiler traces directory: /profiles/$name" "$logs/server.log"; then
      found=true
      break
    fi
    sleep 2
  done
  if [[ "$found" != true ]]; then
    printf 'Profiling startup log unavailable; inspect Modal logs.\n' >&2
    exit 1
  fi

  "$PY" -m bench.profile_capture \
    --manifest "$MANIFEST" \
    --expected-prefix-caching "$expected" \
    --profile-session "$name" \
    --server-log "$logs/server.log" \
    --output-dir "$output"

  mkdir -p "$output/traces"
  verified=false
  for attempt in {1..3}; do
    if "$PY" -m modal volume get "$VOLUME" "$name" "$output/traces" \
      --env main --force && check_profile_traces "$output"; then
      verified=true
      break
    fi
    sleep 5
  done
  if [[ "$verified" != true ]]; then
    printf 'Trace download or GPU-event check failed; preserve this session for inspection.\n' >&2
    exit 1
  fi
)
```

## 4. Run cache-off, then cache-on

Run this as a plain command; let the helper finish and shut down the app:

```bash
capture_profile shared-c8-off-01 off
```

Continue only if it printed a successful GPU-event check and app shutdown. If it failed, inspect that session's logs and `capture.json` before another GPU attempt.

```bash
capture_profile shared-c8-on-01 on shared-c8-off-01
```

The second command checks that its server identity differs from the first. It uses the same manifest and a separate trace directory. Each session makes 17 generation requests: eight warmups, one primer, and eight profiled requests.

## 5. Check the pair and open the traces

```bash
"$PY" - "$PROFILE_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = []
for name, expected in (("shared-c8-off-01", False), ("shared-c8-on-01", True)):
    path = root / name
    run = json.loads((path / "capture.json").read_text(encoding="utf-8"))
    receipt = json.loads((path / "trace-check.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed" and run["profiler"]["stop_completed"]
    assert run["server"]["expected_prefix_caching"] is expected
    results = run["captured_batch"]["results"]
    assert len(results) == 8 and all(result["success"] for result in results)
    assert sum(trace["gpu_kernel_events"] for trace in receipt["traces"]) > 0
    runs.append(run)
    print(name, "hits:", run["metrics"]["prefix_hits_tokens"])
assert runs[0]["manifest_sha256"] == runs[1]["manifest_sha256"]
assert runs[0]["request_ids"] == runs[1]["request_ids"]
assert runs[0]["server"]["process_start_time_seconds"] != runs[1]["server"]["process_start_time_seconds"]
assert runs[0]["metrics"]["prefix_hits_tokens"] == 0
assert runs[1]["metrics"]["prefix_hits_tokens"] > 0
print("Pair validated. Inspect the GPU worker traces next.")
PY
```

Open the GPU-containing `.json.gz` files identified by each `trace-check.json` in [Perfetto](https://ui.perfetto.dev/). vLLM documents this viewer for its profiler traces. Compare the beginning of the batch, GPU kernel activity, and subsequent decode iterations. Keep frontend CPU traces distinct from GPU worker traces. [vLLM profiling](https://docs.vllm.ai/en/v0.21.0/contributing/profiling/)

Retain the raw traces and receipts. Kernel-event presence does not prove that every expected operation was captured, nor does a profile establish an uninstrumented speedup. Keep these captures separate from the published cache-study results. The driver leaves `trace_files_verified` false; the separate receipt records the download check without rewriting the original capture record.

## Recovery and returning to benchmarks

If the shell closes or automatic shutdown reports an error, stop the app explicitly:

```bash
"$PY" -m modal app stop "$APP" --env main --yes
```

Do not delete a failed session or overwrite remote traces during diagnosis. Resolve the cause before choosing fresh labels. If labels change, use the matching labels in the final pair check too.

Before a future normal benchmark deployment, clear any globally exported profiling setting:

```bash
unset INFERENCE_LAB_PROFILE_SESSION
```

The helper's deployment variables apply only to its deploy command. Stopping the profiling app does not restore a normal deployment; the normal benchmark workflow must redeploy its own configuration.
