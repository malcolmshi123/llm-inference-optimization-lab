# Validate the tooling and saved evidence

Run these commands in **Git Bash**, from the repository root, with the dependencies in `requirements.txt` installed. Python 3.11 matches [CPU CI](../.github/workflows/tests.yml). These checks run locally without a Modal deployment, GPU, model weights, or proxy credentials. Only the optional tokenizer download needs network access.

The commands validate existing measurements and regenerate analysis outputs. They do not rerun the performance experiments or establish a new speedup.

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

Keep this terminal open for the remaining commands. Each comparison creates a new output directory and refuses to overwrite it. Repeat this setup to get a fresh temporary directory before rerunning the comparisons. The committed comparisons and raw evidence remain unchanged.

## 2. Run the CPU test suite offline

```bash
HF_HOME="$CHECK_DIR/huggingface" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
INFERENCE_LAB_REQUIRE_TOKENIZER=false \
MPLBACKEND=Agg \
"$PY" -m unittest discover -s tests -v
```

Expect `OK`, with the real-tokenizer check skipped because this temporary Hugging Face cache is empty. The other tests use local fixtures and fake responses; they cover the benchmark, comparisons, server timings, and profiling tools. A passing suite validates the tooling, not GPU performance.

### Optional: require the pinned tokenizer check

To match the separate CI integration step, download only the pinned tokenizer into the same temporary cache. This command requires network access:

```bash
HF_HOME="$CHECK_DIR/huggingface" "$PY" - <<'PY'
from transformers import AutoTokenizer
from server_config import MODEL_NAME, MODEL_REVISION

AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
PY
```

Then run the integration check offline. It must pass rather than skip:

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

Expect both pilots to print `validated`, followed by scenario cache metrics. This checks saved requests, startup evidence, raw metric deltas, matching workloads, distinct server processes, and treatment order. Pilots remain separate from the full measured study.

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

Expect a successful export containing `comparison.json`, `comparison.csv`, and three PNG/SVG figure pairs: `01_cache_latency`, `02_cache_prefill`, and `03_server_timings`. The four sessions contain 512 measured requests, excluding preparation. Validation checks workload pairing, execution order, distinct process identities, request results, raw snapshots, and required cache counters.

The two rounds remain exploratory and are reported separately. Recorded dirty-worktree warnings describe the original captures; this check cannot reconstruct uncommitted source from those sessions.

## 5. Recompute the profiling summary

```bash
MPLBACKEND=Agg "$PY" -m bench.profile_summary \
  --off "$PROFILE_DIR/shared-c8-off-01" \
  --on "$PROFILE_DIR/shared-c8-on-01" \
  --output-dir "$CHECK_DIR/profile-comparison"
```

Expect a successful export containing `summary.json`, `kernels.csv`, `annotations.csv`, and `01_gpu_timeline.png` / `.svg`. This revalidates requests, metric windows, trace hashes and receipts, GPU events, and the paired workload/server identities. Each treatment profiles one eight-request batch; profiling overhead is included. Kernel durations and host annotations do not establish GPU utilization or a prefill/decode GPU-time split.

For new trace captures, use the separate [profiling runbook](profiling.md).

## Interpreting failures

A missing `server.log` or `.prom` file means the saved evidence is incomplete. Restore the original file; do not synthesize startup evidence. The session `server.log` files are intentionally tracked despite the general `*.log` ignore rule.

An existing output directory means the export is protected against overwriting. Repeat section 1 to choose a fresh temporary output directory.

A failed required-tokenizer check means the pinned tokenizer could not be loaded or its length/prefix assertions failed. Check the download step before changing the workload.

Keep the temporary outputs while reviewing them. They live outside the repository and can be deleted afterward. Passing these checks confirms consistency of the saved evidence and tooling; it does not add independent experimental replications.
