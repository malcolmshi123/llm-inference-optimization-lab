# Capture a small profiling pair

Run the profiling pair in Git Bash from the repository root. The procedure uses the existing A10 deployment and captures eight shared-prefix requests per treatment. GPU startup and image preparation may take longer than the capture. The procedure has passed offline checks. Use the first live run to verify the capture on Modal.

Use one terminal for both treatments. Run cache-off first and verify its downloaded trace before starting cache-on. Keep the app idle apart from the capture. Pause the normal benchmark helper and any other clients.

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

Keep credentials in the shell environment. Check that the app name and endpoint refer to the same deployment. If Modal prints a different endpoint, stop and update `VLLM_BASE_URL` before running the capture.

## 2. Define the download check

The check reads each compressed trace and requires at least one GPU kernel event across the download. CPU-only frontend traces can accompany the GPU worker traces. The check writes a separate receipt and leaves `capture.json` intact. The receipt verifies the trace download. An optimization claim needs performance measurements.

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

Each call deploys one treatment, checks readiness and the startup log, captures and downloads traces, then stops the app. The exit trap attempts shutdown on failure or Ctrl+C. Use a fresh label for each attempt, including retries. The helper rejects existing local session or log directories, and unique labels keep remote traces from different attempts apart.

The readiness request times out after 300 seconds. The helper polls for the startup log up to ten times and tries the download up to three times. A failed capture requires a new attempt. Modal may need a few seconds to commit flushed files to the Volume, so each download retry uses the trace directory created for that session. The `--force` flag refreshes the local copies in that directory. [Volume persistence](https://modal.com/docs/guide/volumes#background-commits)

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

Run cache-off as a plain command and wait for the helper to finish and stop the app.

```bash
capture_profile shared-c8-off-01 off
```

Start cache-on after the GPU-event check passes and the app stops. If cache-off fails, inspect the session logs and `capture.json` before starting another GPU run.

```bash
capture_profile shared-c8-on-01 on shared-c8-off-01
```

Cache-on uses the same manifest and a separate trace directory. The helper checks that the server process differs from the cache-off process. Each session sends 17 generation requests. Eight warm up the server, one primes the prefix, and eight run under the profiler.

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

Use each `trace-check.json` to find the `.json.gz` files that contain GPU events. Open those files in [Perfetto](https://ui.perfetto.dev/), the viewer vLLM documents for these traces. Compare the start of the batch and the kernel activity in later decode iterations. Read the frontend CPU traces separately from the GPU worker traces. [vLLM profiling](https://docs.vllm.ai/en/v0.21.0/contributing/profiling/)

Keep the raw traces and receipts with the profiling runs, separate from the published cache-study results. A trace with kernel events may still be missing operations, and a profiled run includes instrumentation overhead. The driver leaves `trace_files_verified` false. Use the separate receipt to confirm the download without editing the original capture record.

## Offline analysis

Once both downloads pass, run the summary on your machine. It needs neither a Modal connection nor a GPU. Pick a fresh output directory because the summary refuses to overwrite existing outputs.

```bash
"$PY" -m bench.profile_summary \
  --off "$PROFILE_DIR/shared-c8-off-01" \
  --on "$PROFILE_DIR/shared-c8-on-01" \
  --output-dir "$PROFILE_DIR/comparison"
```

The export includes `summary.json`, `kernels.csv`, `annotations.csv`, and `01_gpu_timeline.png` / `.svg`. The CSV and JSON keep full kernel names. Before writing, the loader checks the selected requests, raw metric windows, distinct server processes, observed GPU properties, trace hashes, kernel counts in the receipts, and coverage of the downloaded files. It accepts one GPU worker trace per capture and keeps CPU-only frontend traces in the inventory. If new-prefill counters are available, the annotated context token counts must agree with them.

Chrome trace JSON records `ts` and `dur` in microseconds even when `displayTimeUnit` is `ms`. The summarizer converts durations to milliseconds. Adding kernel durations counts concurrent work more than once. The interval union counts overlapping time once on the single GPU. The span runs from the first kernel start to the last kernel end. Measuring SM utilization requires other data. The plot closes gaps shorter than 0.5 ms for display while the export retains the exact sums, spans, and union durations. [Chrome trace format](https://perfetto.dev/docs/getting-started/other-formats)

vLLM 0.21 records context and generation request and token counts for each worker execution. A request counts as context work until it generates its first output token, including later prefill chunks. The summary exports each host scope’s original name and counts and labels it `context_only`, `mixed`, `generation_only`, or `empty`. These scopes mark ranges on the host around model execution. Timestamp overlap alone cannot attribute GPU kernels to either phase. [Worker annotations](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/v1/worker/gpu_worker.py#L682), [iteration counts](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/v1/utils.py#L433)

See the [README case study](../README.md#gpu-profiling-case-study) for the findings. The summary writes its own outputs and leaves the captures, receipts, and original benchmark comparison intact.

## Recovery and returning to benchmarks

If the shell closes or the shutdown trap reports an error, stop the app with this command.

```bash
"$PY" -m modal app stop "$APP" --env main --yes
```

Keep failed sessions and remote traces while you investigate. Fix the cause before trying again with fresh labels. If you change the labels, update them in the final pair check too.

Clear any profiling setting you exported before redeploying for regular benchmarks.

```bash
unset INFERENCE_LAB_PROFILE_SESSION
```

The helper sets deployment variables for its deploy command alone. After stopping the profiling app, redeploy with the normal benchmark workflow to restore its configuration.
