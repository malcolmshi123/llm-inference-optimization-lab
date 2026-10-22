run_treatment() (
  set -euo pipefail

  if (( $# < 3 || $# > 4 )); then
    printf 'Usage: run_treatment NAME off|on WORKLOAD [PREVIOUS]\n' >&2
    exit 1
  fi

  : "${PY:?Set PY to the existing virtual-environment interpreter}"
  : "${APP:?Set APP to the Modal app name}"
  : "${LAB_DIR:?Set LAB_DIR to the study directory}"
  : "${VLLM_BASE_URL:?Set VLLM_BASE_URL to the endpoint root}"
  : "${MODAL_PROXY_KEY:?Set MODAL_PROXY_KEY}"
  : "${MODAL_PROXY_SECRET:?Set MODAL_PROXY_SECRET}"
  test -f modal_app.py
  test -f server_config.py

  name="$1"
  expected="$2"
  workload="$3"
  previous="${4:-}"

  case "$expected" in
    off) cache=false ;;
    on)  cache=true ;;
    *) printf 'Treatment must be off or on\n' >&2; exit 1 ;;
  esac

  if [[ -e "$LAB_DIR/$name" ]]; then
    printf 'Output already exists: %s\nUse a new session name.\n' \
      "$LAB_DIR/$name" >&2
    exit 1
  fi

  test -f "$LAB_DIR/workloads/$workload.json"
  mkdir -p "$LAB_DIR/logs"

  previous_args=()
  if [[ -n "$previous" ]]; then
    test -f "$LAB_DIR/$previous/run.json"
    previous_args=(--previous-run "$LAB_DIR/$previous/run.json")
  fi

  # Stop the deployment on success, failure, or interruption.
  trap '"$PY" -m modal app stop "$APP" --env main --yes' EXIT

  since=$("$PY" -c \
    'from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat())')

  INFERENCE_LAB_PREFIX_CACHING="$cache" \
  INFERENCE_LAB_MAX_NUM_BATCHED_TOKENS=2048 \
  "$PY" -m modal deploy modal_app.py --env main

  "$PY" -m modal function autoscaler update "$APP/serve" \
    --env main \
    --min-containers 1

  curl --location --fail --silent --show-error \
    --max-time 300 \
    -H "Modal-Key: $MODAL_PROXY_KEY" \
    -H "Modal-Secret: $MODAL_PROXY_SECRET" \
    --output "$LAB_DIR/logs/$name-ready.prom" \
    --write-out 'Readiness: HTTP %{http_code}, %{time_total}s, %{size_download} bytes\n' \
    "$VLLM_BASE_URL/metrics"

  if ! grep -q '^process_start_time_seconds ' \
    "$LAB_DIR/logs/$name-ready.prom"; then
    printf 'Readiness response does not contain server metrics.\n' >&2
    exit 1
  fi

  # Allow a short delay for startup logs to become available.
  found=false
  for attempt in {1..15}; do
    "$PY" -m modal app logs "$APP" \
      --env main \
      --since "$since" \
      --source stdout \
      --tail 1000 \
      > "$LAB_DIR/logs/$name.txt"

    if grep -qF "Starting vLLM server with command: vllm serve " \
      "$LAB_DIR/logs/$name.txt"; then
      found=true
      break
    fi

    sleep 2
  done

  if [[ "$found" != true ]]; then
    printf 'Startup log unavailable. Inspect Modal logs before retrying.\n' >&2
    exit 1
  fi

  "$PY" -m bench.cache_study run \
    --manifest "$LAB_DIR/workloads/$workload.json" \
    --expected-prefix-caching "$expected" \
    --server-log "$LAB_DIR/logs/$name.txt" \
    --output-dir "$LAB_DIR/$name" \
    "${previous_args[@]}"

  "$PY" - "$LAB_DIR/$name" "$expected" <<'PY'
import sys
from pathlib import Path
from bench.compare_cache import load_run

run, manifest, rows = load_run(
    Path(sys.argv[1]), expected=sys.argv[2] == "on"
)
print("Validated:", run["status"])
print("Requests per scenario:", manifest["count_per_scenario"])
for row in rows:
    print({
        key: row[key]
        for key in (
            "scenario", "successful", "failed",
            "ttft_p50_ms", "output_tokens_per_s",
            "token_cache_hit_rate", "new_prefill_tokens",
        )
    })
PY
)
