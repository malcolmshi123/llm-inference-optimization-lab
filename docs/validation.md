# Validate the tooling and saved evidence

From the repository root, open Git Bash and install the dependencies in `requirements.txt`. Use Python 3.11 to match [CPU CI](../.github/workflows/tests.yml). You can run the checks locally without a Modal deployment, GPU, model weights, or proxy credentials. The optional tokenizer download is the only step that needs network access.

Use these checks to verify saved measurements and regenerate the analysis outputs. A new speedup claim would require a new performance experiment.

## 1. Set up the shell and temporary outputs

```bash
if [[ -x .venv/Scripts/python.exe ]]; then
  PY="$PWD/.venv/Scripts/python.exe"
else
  PY="$PWD/.venv/bin/python"
fi
test -x "$PY"
test -f modal_app.py

CHECK_DIR=$("$PY" -c \
  'import tempfile; print(tempfile.mkdtemp(prefix="inference-lab-validation-"))')
LAB_DIR="$PWD/inference-runs/cache-study-live-01"
PROFILE_DIR="$PWD/inference-runs/profiling"
printf 'Validation outputs: %s\n' "$CHECK_DIR"
```

Keep this terminal open so the remaining commands can use the paths you just set. The comparison tools require fresh output directories. Repeat this setup before rerunning either comparison. The committed comparisons and raw evidence stay where they are.

## 2. Run the CPU test suite offline

```bash
HF_HOME="$CHECK_DIR/huggingface" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
INFERENCE_LAB_REQUIRE_TOKENIZER=false \
MPLBACKEND=Agg \
"$PY" -m unittest discover -s tests -v
```

Look for `OK`. The real-tokenizer check skips because the temporary Hugging Face cache is empty. The remaining tests use local fixtures and fake responses to exercise the benchmark, comparisons, server timing calculations, and profiling tools. GPU performance requires a live benchmark.

### Optional: require the pinned tokenizer check

Download the pinned tokenizer into the same temporary cache to match the separate CI integration step. Run this command with network access.

```bash
HF_HOME="$CHECK_DIR/huggingface" "$PY" - <<'PY'
from transformers import AutoTokenizer
from server_config import MODEL_NAME, MODEL_REVISION

AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
PY
```

Run the integration check offline. Expect it to pass with the tokenizer available in the cache.

```bash
HF_HOME="$CHECK_DIR/huggingface" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
INFERENCE_LAB_REQUIRE_TOKENIZER=true \
"$PY" -m unittest \
  tests.test_cache_workloads.CacheWorkloadTests.test_cached_model_tokenizer_confirms_real_lengths_and_prefixes \
  -v
```

## 3. Validate the saved pilot pair

```bash
"$PY" -m bench.check_cache_pilots "$LAB_DIR"
```

Each pilot should print `validated` followed by its scenario cache metrics. The check verifies saved requests, startup evidence, raw metric deltas, workload matches, distinct server processes, and treatment order. The full study uses separate measurements.

## 4. Recompute the full cache comparison

```bash
MPLBACKEND=Agg "$PY" -m bench.compare_cache \
  --round-1-off "$LAB_DIR/round-1-off" \
  --round-1-on "$LAB_DIR/round-1-on" \
  --round-2-off "$LAB_DIR/round-2-off" \
  --round-2-on "$LAB_DIR/round-2-on" \
  --output-dir "$CHECK_DIR/cache-comparison" \
  --require-cache-metrics
```

Look for `comparison.json`, `comparison.csv`, and PNG and SVG versions of `01_cache_latency`, `02_cache_prefill`, and `03_server_timings`. The four sessions contain 512 measured requests after excluding preparation. The comparison checks workload pairing, execution order, distinct server processes, request results, raw snapshots, and the required cache counters.

Read the two exploratory rounds separately. A dirty-worktree warning describes the checkout at capture time. The saved measurements cannot recover those uncommitted changes.

## 5. Recompute the profiling summary

```bash
MPLBACKEND=Agg "$PY" -m bench.profile_summary \
  --off "$PROFILE_DIR/shared-c8-off-01" \
  --on "$PROFILE_DIR/shared-c8-on-01" \
  --output-dir "$CHECK_DIR/profile-comparison"
```

Look for `summary.json`, `kernels.csv`, `annotations.csv`, and `01_gpu_timeline.png` / `.svg`. The summary checks the requests, metric windows, trace hashes and receipts, GPU events, and the workloads and server identities for both runs. Each treatment profiles one eight-request batch, so the timings include profiler overhead. Kernel durations and host annotations alone cannot tell you GPU utilization or how GPU time splits between prefill and decode.

To capture new traces, follow the [profiling runbook](profiling.md).

## Interpreting failures

If a `server.log` or `.prom` file is missing, restore the original to complete the evidence. Use only the startup log from that session. Git tracks the session `server.log` files as exceptions to the `*.log` ignore rule.

If the output directory already exists, repeat section 1 to create a fresh temporary directory.

If the required-tokenizer check fails, confirm the pinned tokenizer downloaded and loads from the temporary cache before changing the workload. If it loads, inspect the length and prefix assertions.

Keep the outputs until you finish reviewing them, then delete the temporary directory outside the repository if you no longer need it. A pass confirms that the tooling and saved evidence agree. Independent replication requires new runs.
