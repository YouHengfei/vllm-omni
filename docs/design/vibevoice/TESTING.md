# VibeVoice Development-Machine Test Runbook

> **Branch:** `fix/vibevoice-review-remediation`
>
> **Preserved comparison branch:** `feat/vibevoice-support` at `e0290cbc`
>
> **Status:** Commands are maintained here; GPU and full pytest execution is
> performed on the development machine.

## 1. Purpose

This runbook is the single launch checklist for validating VibeVoice remediation
work on a machine with the complete vLLM-Omni environment and NVIDIA GPUs. It
separates static/CPU validation from real GPU correctness, quality, performance,
and stability validation.

Do not report a test as passed from collection, compilation, mocks, or a machine
that lacks the required hardware. Record the exact command, dependency versions,
selected/deselected/skipped counts, and failure output.

## 2. Tonight's execution order

Use one artifact directory for every command and do not optimize or edit runtime
code between comparison runs:

```bash
set -e -o pipefail
export VIBEVOICE_RESULT_ROOT="/tmp/vibevoice-validation-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${VIBEVOICE_RESULT_ROOT}"
exec > >(tee -a "${VIBEVOICE_RESULT_ROOT}/session.log") 2>&1
```

Run in this order:

1. update and verify the checkout (Section 3);
2. record hardware, dependency, checkpoint, and config-schema identity
   (Section 4);
3. run static, CPU, and collection gates (Sections 5–10);
4. run focused GPU model-executor tests (Section 11);
5. run the official-weight suite once with the **full-eager** overlay, then once
   with the checked-in default graph configuration (Section 12);
6. run raw/SSE, abort, independent 2/3/4/5-token completion, and four-tab
   browser checks (Sections 12–14);
7. only after all correctness gates pass, run quality, the five-cell graph
   performance matrix, and one-hour stability (Sections 15–17);
8. copy the completed report template and all artifact paths into the feedback
   message (Section 18).

Stop immediately and preserve logs if any of these occur:

- the runtime config class/schema differs from the recorded expected contract;
- any SSE stream emits `speech.audio.error`, ends without one
  `speech.audio.done`, or emits audio after done;
- an eligible graph capture fails while
  `cuda_graph_capture_failure_fatal=true`;
- audio sample count differs from `max_new_tokens * 3,200` for a length-capped
  request;
- VRAM grows monotonically across completed/aborted requests;
- the post-abort health probe fails.

A failed gate is useful evidence. Do not work around it by loosening an
assertion, disabling strict graph capture, or substituting bundled reference
voices.

## 3. Update the development-machine checkout

```bash
git fetch origin
git switch fix/vibevoice-review-remediation
git pull --ff-only origin fix/vibevoice-review-remediation

git status --short --branch
git rev-parse HEAD
git rev-parse origin/fix/vibevoice-review-remediation
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/fix/vibevoice-review-remediation)"
```

The known-working implementation remains available for comparison. Prefer a
separate detached worktree so the preserved branch cannot be changed by test
artifacts:

```bash
git worktree add ../vllm-omni-vibevoice-baseline \
  e0290cbc33054596380565ad56fd65aa934dd69e
```

Never force-push `feat/vibevoice-support`.

## 4. Record the environment and schema identity

Run and save the output before testing:

```bash
nvidia-smi
python --version
python - <<'PY'
import diffusers
import torch
import transformers
import vllm
import vllm_omni

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
print("vllm:", vllm.__version__)
print("vllm_omni:", getattr(vllm_omni, "__version__", "unknown"))
print("transformers:", transformers.__version__)
print("diffusers:", diffusers.__version__)
PY

git rev-parse HEAD
git submodule status || true
```

Record the exact model revisions or local checkpoint paths as well:

```bash
export VIBEVOICE_TEST_MODEL=microsoft/VibeVoice-1.5B
export VIBEVOICE_TEST_TOKENIZER=Qwen/Qwen2.5-1.5B
export VIBEVOICE_TEST_MODEL_REVISION=REPLACE_WITH_IMMUTABLE_SHA
export VIBEVOICE_TEST_TOKENIZER_REVISION=REPLACE_WITH_IMMUTABLE_SHA
export VIBEVOICE_OFFICIAL_REPO=/path/to/Microsoft/VibeVoice
```

Resolve and record immutable Hub revisions before replacing the placeholders.
For local checkpoints, the script records the path and `config.json` hash:

```bash
python - <<'PY'
import hashlib
import os
from pathlib import Path

from huggingface_hub import HfApi

for value, revision in (
    (os.environ["VIBEVOICE_TEST_MODEL"], os.getenv("VIBEVOICE_TEST_MODEL_REVISION")),
    (os.environ["VIBEVOICE_TEST_TOKENIZER"], os.getenv("VIBEVOICE_TEST_TOKENIZER_REVISION")),
):
    path = Path(value).expanduser()
    if path.exists():
        config = path / "config.json"
        digest = hashlib.sha256(config.read_bytes()).hexdigest() if config.exists() else "no-config"
        print(value, "local_path=", path.resolve(), "config_sha256=", digest)
    else:
        info = HfApi().model_info(value, revision=revision)
        print(value, "requested_revision=", revision, "resolved_sha=", info.sha)
PY
```

Then verify the config class and normalized flat schema used by this branch:

```bash
python - <<'PY'
import os

from transformers import AutoConfig
from vllm_omni.engine.arg_utils import _register_omni_hf_configs

_register_omni_hf_configs()
config = AutoConfig.from_pretrained(
    os.environ["VIBEVOICE_TEST_MODEL"],
    revision=os.getenv("VIBEVOICE_TEST_MODEL_REVISION"),
    trust_remote_code=False,
)
print("config_class:", type(config).__module__, type(config).__qualname__)
for name in (
    "audio_config",
    "semantic_model_config",
    "text_config",
    "num_head_layers",
    "ddpm_num_steps",
    "ddpm_num_inference_steps",
):
    print(name, "=", getattr(config, name, "<missing>"))
assert all(
    hasattr(config, name)
    for name in (
        "audio_config",
        "semantic_model_config",
        "text_config",
        "num_head_layers",
        "ddpm_num_steps",
        "ddpm_num_inference_steps",
    )
), "Unexpected VibeVoice config schema; stop and report before GPU testing."
PY
```

For an offline environment, point the model/tokenizer variables to immutable
local snapshot directories and unset the two revision variables. Do not test a
mutable Hub branch without recording its resolved SHA.

## 5. Static gates

Use the repository's locked tools, not an arbitrary globally installed Ruff:

```bash
git diff --check

changed_python_files=$(git diff --name-only 67c54777...HEAD -- '*.py')
uvx ruff@0.14.10 check ${changed_python_files}
uvx ruff@0.14.10 format --check ${changed_python_files}

python -m compileall -q vllm_omni tests examples
python tools/pre_commit/check_test_marks.py \
  $(git diff --name-only 67c54777...HEAD -- 'tests/**/*.py')

pre-commit run --from-ref 67c54777 --to-ref HEAD

python - <<'PY'
from pathlib import Path
from docs.mkdocs.hooks.generate_examples import Example

source = Path("examples/online_serving/text_to_speech").resolve()
rendered = Example(source, "online_serving").generate()
for required in (
    "## VibeVoice",
    "text_to_speech/vibevoice/app.py",
    "text_to_speech/vibevoice/pcm-player-worklet.js",
):
    assert required in rendered, required
PY

DOCS_WORKTREE="${VIBEVOICE_RESULT_ROOT}/docs-worktree"
git worktree add --detach "${DOCS_WORKTREE}" HEAD
(
  cd "${DOCS_WORKTREE}"
  mkdocs build --strict --site-dir "${VIBEVOICE_RESULT_ROOT}/site"
)
git worktree remove --force "${DOCS_WORKTREE}"

python .buildkite/common/scripts/upload_pipeline.py \
  .buildkite/cuda/test-merge.yml \
  > "${VIBEVOICE_RESULT_ROOT}/buildkite-diff-aware.yml"
python .buildkite/common/scripts/upload_pipeline.py --all \
  .buildkite/cuda/test-merge.yml \
  > "${VIBEVOICE_RESULT_ROOT}/buildkite-all.yml"
rg -n 'TTS · VibeVoice Test' "${VIBEVOICE_RESULT_ROOT}"/buildkite-*.yml
```

After rebasing onto a newer upstream base, replace `67c54777` with the locked
merge-base recorded in `README.md`.

## 6. M1 CPU verification

Run the M1 behavior-preserving hygiene tests first:

```bash
pytest -q -s \
  tests/model_executor/models/vibevoice/test_vibevoice_config.py \
  tests/model_executor/models/vibevoice/test_vibevoice_processing.py \
  tests/model_executor/models/vibevoice/test_vibevoice_adapter.py \
  tests/dfx/test_stability_helpers.py \
  tests/e2e/accuracy/vibevoice/test_vibevoice_quality.py \
  -m 'core_model and cpu' \
  --run-level core_model
```

Run the tokenizer resolver tests explicitly so a broad marker selection cannot
hide them:

```bash
pytest -q -s \
  tests/model_executor/models/vibevoice/test_vibevoice_config.py \
  -k 'vibevoice_tokenizer_contract or vibevoice_tokenizer_fallback' \
  --run-level core_model
```

Expected M1 properties:

- local `preprocessor_config.json` resolves the tokenizer contract;
- remote metadata is read through vLLM's `get_hf_file_to_dict` helper;
- `revision` is forwarded;
- missing metadata or a download failure returns `None`;
- malformed/non-object metadata fails clearly;
- no test requires a GPU or model weights.

## 7. M2 CPU verification

Run the Delta transport and request-lifecycle contracts:

```bash
pytest -q -s \
  tests/model_executor/models/vibevoice/test_vibevoice_stateful.py \
  tests/model_executor/models/vibevoice/test_vibevoice_adapter.py \
  tests/worker/test_gpu_ar_model_runner.py \
  tests/worker/test_omni_gpu_model_runner.py \
  tests/entrypoints/openai_api/test_serving_speech_stream.py \
  tests/engine/test_output_processor.py \
  -m 'core_model and cpu' \
  --run-level core_model
```

Required M2 properties:

- marked Delta list snapshots emit every current chunk exactly once;
- unmarked cumulative lists retain legacy tail behavior;
- raw PCM requests pass adapter validation;
- internal state uses `_omni_req_id`, not user-controlled `request_id`;
- unscheduled finishes clean immediately and scheduled finishes survive through
  final postprocess;
- zero-token callbacks carry no scheduled request IDs;
- pending waveform events are synchronized before state is dropped;
- model state clears before the named-KV branch closes at shutdown.

## 8. M3A control and graph-policy verification

Run the strict request-control, deployment-schema, and finite graph-policy
contracts:

```bash
pytest -q -s \
  tests/model_executor/models/vibevoice/test_vibevoice_adapter.py \
  tests/model_executor/models/vibevoice/test_vibevoice_stateful.py \
  tests/model_executor/models/vibevoice/test_vibevoice_named_kv_branch.py \
  tests/model_executor/models/vibevoice/test_vibevoice_diffusion.py \
  tests/model_executor/models/vibevoice/test_vibevoice_config.py \
  -m 'core_model and cpu' \
  --run-level core_model
```

Required M3A CPU properties:

- guidance is a finite JSON number in `[0.0, 20.0]`;
- diffusion steps are an integer in `[1, 50]`, without string, float, or bool
  coercion;
- unknown request controls fail before request state is created;
- a failed multi-control update leaves existing request controls unchanged;
- unknown or malformed `vibevoice_runtime_config` fails startup;
- deploy defaults are guidance 1.3, 10 steps, and at most four graph batches;
- the public graph policy has exactly four eligible control keys.

Run the focused CUDA Graph gate on one supported GPU:

```bash
VIBEVOICE_TEST_MODEL=${VIBEVOICE_TEST_MODEL} \
pytest -q -s \
  tests/model_executor/models/vibevoice/test_vibevoice_diffusion_graph_gpu.py \
  -m 'core_model and cuda' \
  --run-level core_model
```

Required M3A GPU properties:

- official controls replay bitwise equal to eager for batches 1, 2, and 4;
- capturing batches 1 through 4 produces exactly four graph entries;
- custom valid guidance/step combinations and batch size 5 return to eager;
- custom controls do not grow or disable the official graph cache;
- a subsequent official-control request still replays successfully.

Record `torch.cuda.max_memory_allocated()` and
`torch.cuda.max_memory_reserved()` before capture, after four official keys,
and after all custom-control probes. Custom probes must not increase the graph
entry count or produce monotonic graph-owned growth.

## 9. M3C–M6 remediation batch

Run the decode-output, generic Adapter hook, named-KV lifecycle, and CPU token
metadata contracts:

```bash
pytest -q -s \
  tests/model_executor/models/vibevoice/test_vibevoice_audio_decode.py \
  tests/model_executor/models/vibevoice/test_vibevoice_stateful.py \
  tests/model_executor/models/vibevoice/test_vibevoice_adapter.py \
  tests/model_executor/models/vibevoice/test_vibevoice_named_kv_branch.py \
  tests/entrypoints/openai_api/test_tts_adapter.py \
  tests/entrypoints/openai_api/test_serving_speech.py \
  tests/entrypoints/openai_api/test_serving_speech_stream.py \
  tests/benchmarks/patch/test_patch.py \
  tests/examples/test_vibevoice_browser_example.py \
  tests/worker/test_gpu_ar_model_runner.py \
  tests/worker/test_omni_gpu_model_runner.py \
  -m 'core_model and cpu' \
  --run-level core_model
```

Required CPU properties:

- default Adapter finalization and sampling hooks are identity operations;
- VibeVoice UUID finalization, token gate, max-token override, finish-header,
  and WebSocket PCM policies are selected through the Adapter interface;
- stage-key lookup resolves VibeVoice and rejects ambiguous registration;
- legacy and schedule-aware finish hooks retain their respective call shapes;
- model state clears before named-KV close, and cleanup failures do not prevent
  upstream shutdown;
- binding failure closes an unpublished named branch without masking the
  original bind error if close also fails;
- SSE benchmark/browser consumers fail on `speech.audio.error` or missing done;
- runtime capacity fields reject fractional and infinite numbers;
- browser proxy connection failure closes its HTTP client;
- VibeVoice control transitions use the scheduler-maintained CPU token span,
  reject missing/misaligned metadata, and never read a GPU token with `.item()`.

Run the focused CUDA contracts:

```bash
VIBEVOICE_TEST_MODEL=${VIBEVOICE_TEST_MODEL} \
pytest -q -s \
  tests/model_executor/models/vibevoice/test_vibevoice_decode_graph_gpu.py \
  tests/model_executor/models/vibevoice/test_vibevoice_diffusion_graph_gpu.py \
  tests/model_executor/models/vibevoice/test_vibevoice_request_cleanup_gpu.py \
  tests/model_executor/models/vibevoice/test_vibevoice_negative_kv_conformance_gpu.py \
  -m 'core_model and cuda' \
  --run-level core_model
```

Required CUDA properties:

- decode graph returns Tensor-valued `semantic_latent` and matches eager across
  consecutive tokens and segment reset;
- diffusion graph remains bounded to the four official-control batch keys;
- injected graph capture failures fall back by default and fail visibly when
  strict capture is requested;
- pending waveform D2H cleanup and positive/negative KV isolation pass;
- a profiler trace contains no per-token synchronization caused by VibeVoice
  control-token inspection.

The ready pipeline's existing `Engine&Model Executor Test` selects
`tests/model_executor -m 'core_model and cuda'` on L4. The merge pipeline adds a
source-gated `TTS · VibeVoice Test` on one H100 for the official-weight offline
and online files. Record Buildkite job URLs and exact selected/skipped counts;
YAML parsing or local collection is not CI execution.

## 10. Full VibeVoice CPU suite

```bash
pytest -q -s tests/model_executor/models/vibevoice \
  -m 'core_model and cpu' \
  --run-level core_model

pytest -q -s \
  tests/entrypoints/openai_api/test_tts_adapter.py \
  tests/entrypoints/openai_api/test_serving_speech_stream.py \
  tests/benchmarks/patch/test_patch.py \
  tests/examples/test_vibevoice_browser_example.py \
  tests/dfx/test_stability_helpers.py \
  tests/e2e/accuracy/vibevoice/test_vibevoice_quality.py \
  -m 'core_model and cpu' \
  --run-level core_model
```

Parity tests that require a local Microsoft source checkout may skip when
`VIBEVOICE_OFFICIAL_REPO` is unset. Record such skips; do not silently count them
as parity passes.

## 11. GPU model-executor tests

Run on one supported NVIDIA GPU unless the test itself requests more:

```bash
VIBEVOICE_TEST_MODEL=${VIBEVOICE_TEST_MODEL} \
pytest -q -s tests/model_executor/models/vibevoice \
  -m 'core_model and cuda' \
  --run-level core_model
```

The required GPU cases include:

- convolution-cache reset between segments/requests;
- diffusion CUDA graph versus eager parity;
- waveform decode CUDA graph versus eager parity;
- named negative-KV branch conformance;
- pending waveform D2H cleanup on request clear;
- finish, abort, failure, and shutdown cleanup;
- the M3A diffusion graph cache remains bounded to four official-control keys.

## 12. Official-weight offline and online E2E

These tests target H100 and advanced level. Use the two checked-in test
overlays. `full_eager.yaml` disables the positive vLLM graph and both VibeVoice
side graphs; `graph_strict.yaml` keeps the checked-in graph defaults but turns
capture fallback into a test failure.

```bash
export VIBEVOICE_FULL_EAGER_CONFIG="$(realpath tests/e2e/vibevoice_configs/full_eager.yaml)"
export VIBEVOICE_GRAPH_STRICT_CONFIG="$(realpath tests/e2e/vibevoice_configs/graph_strict.yaml)"
cp "${VIBEVOICE_FULL_EAGER_CONFIG}" "${VIBEVOICE_RESULT_ROOT}/"
cp "${VIBEVOICE_GRAPH_STRICT_CONFIG}" "${VIBEVOICE_RESULT_ROOT}/"
```

Run the complete suite against full eager first:

```bash
export VIBEVOICE_TEST_DEPLOY_CONFIG="${VIBEVOICE_FULL_EAGER_CONFIG}"
pytest -q -s \
  tests/e2e/offline_inference/test_vibevoice_tts.py \
  tests/e2e/online_serving/test_vibevoice_tts.py \
  -m 'advanced_model and cuda and H100' \
  --run-level advanced_model 2>&1 | \
  tee "${VIBEVOICE_RESULT_ROOT}/e2e-full-eager.log"
```

Then run the same suite with strict graph capture. The server log must contain
successful diffusion/decode capture messages and no fallback warning:

```bash
export VIBEVOICE_TEST_DEPLOY_CONFIG="${VIBEVOICE_GRAPH_STRICT_CONFIG}"
pytest -q -s \
  tests/e2e/offline_inference/test_vibevoice_tts.py \
  tests/e2e/online_serving/test_vibevoice_tts.py \
  -m 'advanced_model and cuda and H100' \
  --run-level advanced_model 2>&1 | \
  tee "${VIBEVOICE_RESULT_ROOT}/e2e-graph-strict.log"

rg -n 'Captured VibeVoice diffusion CUDA graph' \
  "${VIBEVOICE_RESULT_ROOT}/e2e-graph-strict.log"
rg -n 'Captured VibeVoice decode CUDA graph' \
  "${VIBEVOICE_RESULT_ROOT}/e2e-graph-strict.log"
! rg -n 'capture failed|falling back to eager' \
  "${VIBEVOICE_RESULT_ROOT}/e2e-graph-strict.log"
unset VIBEVOICE_TEST_DEPLOY_CONFIG
```

Record independently:

- single request result;
- four-request batch result;
- SSE result with exactly two 3,200-sample deltas and one terminal done;
- raw PCM streaming result;
- uploaded voice/reference audio result;
- concurrent requests independently capped at 2, 3, 4, and 5 tokens;
- aborted request followed by a healthy bounded probe;
- concurrent official-graph and custom-eager controls followed by a healthy
  post-abort probe.

## 13. Manual server launch

### TP=1

Build optional revision arguments once; local snapshot paths intentionally leave
the corresponding arrays empty:

```bash
MODEL_REVISION_ARGS=()
TOKENIZER_REVISION_ARGS=()
if [[ -n "${VIBEVOICE_TEST_MODEL_REVISION:-}" ]]; then
  MODEL_REVISION_ARGS=(--revision "${VIBEVOICE_TEST_MODEL_REVISION}")
fi
if [[ -n "${VIBEVOICE_TEST_TOKENIZER_REVISION:-}" ]]; then
  TOKENIZER_REVISION_ARGS=(--tokenizer-revision "${VIBEVOICE_TEST_TOKENIZER_REVISION}")
fi

vllm serve "${VIBEVOICE_TEST_MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port 8000 \
  "${MODEL_REVISION_ARGS[@]}" \
  --tokenizer "${VIBEVOICE_TEST_TOKENIZER}" \
  "${TOKENIZER_REVISION_ARGS[@]}"
```

With server-local reference files:

```bash
vllm serve "${VIBEVOICE_TEST_MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port 8000 \
  "${MODEL_REVISION_ARGS[@]}" \
  --tokenizer "${VIBEVOICE_TEST_TOKENIZER}" \
  "${TOKENIZER_REVISION_ARGS[@]}" \
  --allowed-local-media-path /path/to/reference-audio
```

### TP=2 (experimental and unverified)

Do not describe TP=2 as supported. Only run after the TP=1 gates pass and the
TP=2 deployment overlay, rank-synchronized diffusion RNG, and acceptance test
are reviewed:

```bash
vllm serve ${VIBEVOICE_TEST_MODEL} \
  --omni \
  --tokenizer ${VIBEVOICE_TEST_TOKENIZER} \
  --deploy-config /path/to/vibevoice-tp2.yaml
```

A successful server start is not sufficient. Generate audio and record both
rank logs, waveform checks, finish reason, peak memory, and latency.

## 14. Streaming and browser validation

M2 defines the raw PCM and Delta SSE contracts. Validate both transports:

1. SSE emits at least two `speech.audio.delta` events before
   `speech.audio.done`.
2. Each audio-token transition contributes exactly 3,200 new samples.
3. Concatenation has no repeated prefix or missing chunk.
4. Raw PCM begins playback before request completion.
5. Audio is mono signed 16-bit PCM at 24 kHz on the wire.
6. Four simultaneous browser requests do not cross speakers or output chunks.

Use the checked-in example:

```bash
python examples/online_serving/text_to_speech/vibevoice/app.py \
  --upstream http://127.0.0.1:8000 \
  --port 7860
```

Follow the VibeVoice section in
`examples/online_serving/text_to_speech/README.md`. Also force one server-side
SSE generation error or interrupt the response: the player must show an error,
not `complete`. Confirm `sseDoneEvents == 1`, `overflowSamples == 0`, and that
sample counters remain request-local after abort/restart. Save browser console
logs, server logs, and one generated WAV/PCM capture per request. This remains
a manual result and must not be reported as pytest coverage.

## 15. Quality and speaker conditioning

Start the server first. Establish the preserved-branch and remediation
baselines with three complete runs per locale. With `--min-evaluated` omitted,
the driver now requires all `--num-prompts` samples to be scored:

```bash
for locale in en zh; do
  for repeat in 1 2 3; do
    python tests/e2e/accuracy/vibevoice/run_vibevoice_quality.py \
      --host 127.0.0.1 \
      --port 8000 \
      --model "${VIBEVOICE_TEST_MODEL}" \
      --tokenizer "${VIBEVOICE_TEST_TOKENIZER}" \
      --locale "${locale}" \
      --num-prompts 8 \
      --max-concurrency 4 \
      --result-dir "${VIBEVOICE_RESULT_ROOT}/quality/${locale}/run-${repeat}" \
      --save-audio-dir "${VIBEVOICE_RESULT_ROOT}/quality/${locale}/run-${repeat}/audio"
  done
done
```

A threshold-free run validates request, PCM, judge, similarity, and terminal
completeness, but is baseline evidence rather than a merge gate. Once baseline
variance is known, run EN and ZH separately with approved thresholds:

```bash
export VIBEVOICE_MAX_MEAN_WER=REPLACE_WITH_APPROVED_EN_THRESHOLD
export VIBEVOICE_MAX_MEAN_CER=REPLACE_WITH_APPROVED_ZH_THRESHOLD
export VIBEVOICE_MIN_MEAN_SIM=REPLACE_WITH_APPROVED_THRESHOLD

python tests/e2e/accuracy/vibevoice/run_vibevoice_quality.py \
  --host 127.0.0.1 --port 8000 \
  --model "${VIBEVOICE_TEST_MODEL}" --tokenizer "${VIBEVOICE_TEST_TOKENIZER}" \
  --locale en --num-prompts 8 --max-concurrency 4 \
  --result-dir "${VIBEVOICE_RESULT_ROOT}/quality-gate/en" \
  --save-audio-dir "${VIBEVOICE_RESULT_ROOT}/quality-gate/en/audio" \
  --max-mean-content-error "${VIBEVOICE_MAX_MEAN_WER}" \
  --min-mean-sim "${VIBEVOICE_MIN_MEAN_SIM}"

python tests/e2e/accuracy/vibevoice/run_vibevoice_quality.py \
  --host 127.0.0.1 --port 8000 \
  --model "${VIBEVOICE_TEST_MODEL}" --tokenizer "${VIBEVOICE_TEST_TOKENIZER}" \
  --locale zh --num-prompts 8 --max-concurrency 4 \
  --result-dir "${VIBEVOICE_RESULT_ROOT}/quality-gate/zh" \
  --save-audio-dir "${VIBEVOICE_RESULT_ROOT}/quality-gate/zh/audio" \
  --max-mean-content-error "${VIBEVOICE_MAX_MEAN_CER}" \
  --min-mean-sim "${VIBEVOICE_MIN_MEAN_SIM}"
```

Record:

- WER/CER mean and per-sample values;
- speaker similarity mean and per-sample values;
- number of natural-stop versus length-capped samples;
- failed/skipped sample count;
- exact threshold values and preserved-baseline comparison.

For speaker conditioning, synthesize identical text from at least two distinct
reference speakers and verify a measured self-reference similarity margin.

## 16. Performance

Compare the same commit/config in this five-cell matrix. The first row is the
only true all-eager baseline; disabling only the two side graphs still leaves
the positive vLLM graph enabled.

| Variant | `enforce_eager` | diffusion graph | decode graph | fatal capture | Checked-in overlay |
| --- | --- | --- | --- | --- | --- |
| full eager | `true` | `false` | `false` | `false` | `full_eager.yaml` |
| positive graph only | `false` | `false` | `false` | `false` | `positive_graph_only.yaml` |
| positive + diffusion | `false` | `true` | `false` | `true` | `diffusion_graph_strict.yaml` |
| positive + decode | `false` | `false` | `true` | `true` | `decode_graph_strict.yaml` |
| positive + both | `false` | `true` | `true` | `true` | `graph_strict.yaml` |

All overlays live in `tests/e2e/vibevoice_configs/` and preserve the checked-in
KV capacity contract through `base_config`. Graph cells fail rather than
silently relabeling eager fallback as graph evidence; retain capture-success
logs with every benchmark result.

For each overlay, start one server in terminal A and retain its log:

```bash
export VARIANT=full_eager  # change for each table row
export OVERLAY="$(realpath tests/e2e/vibevoice_configs/${VARIANT}.yaml)"
MODEL_REVISION_ARGS=()
TOKENIZER_REVISION_ARGS=()
if [[ -n "${VIBEVOICE_TEST_MODEL_REVISION:-}" ]]; then
  MODEL_REVISION_ARGS=(--revision "${VIBEVOICE_TEST_MODEL_REVISION}")
fi
if [[ -n "${VIBEVOICE_TEST_TOKENIZER_REVISION:-}" ]]; then
  TOKENIZER_REVISION_ARGS=(--tokenizer-revision "${VIBEVOICE_TEST_TOKENIZER_REVISION}")
fi

vllm serve "${VIBEVOICE_TEST_MODEL}" \
  --omni \
  --host 127.0.0.1 --port 8000 \
  "${MODEL_REVISION_ARGS[@]}" \
  --tokenizer "${VIBEVOICE_TEST_TOKENIZER}" \
  "${TOKENIZER_REVISION_ARGS[@]}" \
  --deploy-config "${OVERLAY}" \
  --allowed-local-media-path "$(realpath tests/assets)" 2>&1 | \
  tee "${VIBEVOICE_RESULT_ROOT}/perf-${VARIANT}-server.log"
```

After `/health` returns 200, run terminal B. This fixed-length request shape
uses an explicit repository reference and saves one JSON file per
variant/concurrency/repetition:

```bash
export VARIANT=full_eager  # must match terminal A
export VIBEVOICE_PERF_REF="$(python - <<'PY'
from pathlib import Path
print(Path('tests/assets/qwen3_tts/clone_2.wav').resolve().as_uri())
PY
)"
export VIBEVOICE_PERF_EXTRA_BODY="$(python - <<'PY'
import json
import os
print(json.dumps({
    'stream': True,
    'stream_format': 'sse',
    'response_format': 'pcm',
    'max_new_tokens': 128,
    'ref_audio': os.environ['VIBEVOICE_PERF_REF'],
}, separators=(',', ':')))
PY
)"

for concurrency in 1 2 4; do
  for repetition in 1 2 3; do
    result_dir="${VIBEVOICE_RESULT_ROOT}/performance/${VARIANT}/c${concurrency}/r${repetition}"
    mkdir -p "${result_dir}"
    vllm bench serve --omni \
      --host 127.0.0.1 --port 8000 \
      --model "${VIBEVOICE_TEST_MODEL}" \
      --tokenizer "${VIBEVOICE_TEST_TOKENIZER}" \
      --endpoint /v1/audio/speech \
      --backend openai-audio-speech \
      --dataset-name random \
      --random-input-len 128 \
      --random-output-len 128 \
      --random-range-ratio 0.0 \
      --num-prompts "$((20 * concurrency))" \
      --max-concurrency "${concurrency}" \
      --request-rate inf \
      --ignore-eos \
      --extra-body "${VIBEVOICE_PERF_EXTRA_BODY}" \
      --percentile-metrics e2el,audio_ttfp,audio_rtf,audio_duration,audio_underrun \
      --num-warmups 2 \
      --save-result \
      --result-dir "${result_dir}" \
      --result-filename result.json
  done
done
```

Stop terminal A cleanly, change `VARIANT` to each remaining overlay stem, and
repeat. Run concurrency 1, 2, and 4. Separate first-request graph capture from warm
replay, use identical prompts/references and generation controls, and run the
preserved `e0290cbc` worktree with the same external benchmark settings.

Record at least:

- audio TTFP;
- E2E latency;
- audio RTF;
- request/audio throughput;
- P50/P95;
- peak allocated and reserved VRAM;
- graph cache entry count and memory behavior;
- warmup plus at least three measured repetitions.

Use the same `vllm bench serve --omni` request shape recorded by the checked-in
stability configuration, shorten the duration for performance runs, and save
one JSON result per matrix cell/repetition. Do not remove the remaining hidden-
state clone or change graph defaults until these measurements justify it.
Commit messages alone are not benchmark evidence.

## 17. Stability

Run the checked-in one-hour configuration after all shorter gates pass. The
test now issues a final `/health` and exact two-token speech probe before its
server fixture is torn down. Monitor VRAM externally at one-minute intervals:

```bash
export VIBEVOICE_GRAPH_STRICT_CONFIG="${VIBEVOICE_GRAPH_STRICT_CONFIG:-$(realpath tests/e2e/vibevoice_configs/graph_strict.yaml)}"
export VIBEVOICE_TEST_DEPLOY_CONFIG="${VIBEVOICE_GRAPH_STRICT_CONFIG}"
export VIBEVOICE_STABILITY_RESULT_DIR="${VIBEVOICE_RESULT_ROOT}/stability"

(
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader
    sleep 60
  done
) > "${VIBEVOICE_RESULT_ROOT}/stability-vram.csv" &
VRAM_MONITOR_PID=$!
trap 'kill ${VRAM_MONITOR_PID} 2>/dev/null || true' EXIT

pytest -q -s tests/dfx/stability/scripts/test_stability_vibevoice.py \
  --test-config-file tests/dfx/stability/tests/test_vibevoice.json 2>&1 | \
  tee "${VIBEVOICE_RESULT_ROOT}/stability.log"

kill ${VRAM_MONITOR_PID} 2>/dev/null || true
wait ${VRAM_MONITOR_PID} 2>/dev/null || true
trap - EXIT
unset VIBEVOICE_TEST_DEPLOY_CONFIG
```

Verify:

- max concurrency is at least 4;
- failed requests remain zero;
- state/KV/graph memory does not grow monotonically;
- the process remains alive and accepts a final probe request;
- generated audio remains finite and request-owned;
- no speaker or waveform chunks cross request IDs.

## 18. Result report template

```markdown
### VibeVoice validation — <commit>

- Date:
- Branch/commit:
- GPU model/count:
- Driver/CUDA/PyTorch:
- vLLM/vLLM-Omni/Transformers/Diffusers:
- Model/tokenizer requested revision and resolved SHA:
- Runtime config class/module and schema fields:
- Deploy overlay(s):

| Suite | Command | Collected | Passed | Failed | Skipped | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Static | ... | — | — | — | — | PASS/FAIL |
| M1 CPU | ... | | | | | PASS/FAIL |
| Full CPU | ... | | | | | PASS/FAIL |
| GPU unit | ... | | | | | PASS/FAIL |
| Full-eager offline/online E2E | ... | | | | | PASS/FAIL |
| Graph-strict offline/online E2E | ... | | | | | PASS/FAIL |
| Quality | ... | | | | | PASS/FAIL |
| Performance | ... | | | | | PASS/FAIL |
| Stability | ... | | | | | PASS/FAIL |
| Browser | manual protocol | — | — | — | — | PASS/FAIL |

- SSE delta sizes/order/done/error result:
- Independent 2/3/4/5-token sample counts:
- Graph capture-success/fallback log result:
- Browser raw/SSE/abort/four-tab counters:
- Quality metrics, repetitions, variance, and thresholds:
- Performance metrics by five graph cells and concurrency:
- Peak VRAM and one-hour VRAM trend:
- Artifacts/log locations:
- Failures or unexpected skips:
```
