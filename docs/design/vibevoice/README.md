# VibeVoice Review and Remediation Plan

> **Status:** `STATIC_VERIFIED | DEV_MACHINE_TEST_PENDING | CI_PENDING |
> BROWSER_PENDING | ASSET_BLOCKED`
>
> **Original implementation branch:** `feat/vibevoice-support`
>
> **Preserved implementation commit:** `e0290cbc33054596380565ad56fd65aa934dd69e`
>
> **Remediation branch:** `fix/vibevoice-review-remediation`
>
> **Original review range:** `67c54777...e0290cbc` (`cadf0a60` included)
>
> **Last updated:** 2026-08-22

## 1. Purpose

This README is the source of truth for reviewing, planning, implementing, and
verifying the VibeVoice remediation work. It records:

- the immutable baseline that is known to pass stress testing on the development
  machine;
- all findings from the original code review;
- the proposed module-by-module remediation sequence;
- decisions, implementation status, test commands, results, and remaining GPU
  verification;
- the final community contribution gates.

This is a working design/remediation document, not the final user-facing model
guide. Before an upstream pull request is submitted, permanent architecture
content should be retained under `docs/design/`, user behavior should be moved
to `docs/models/vibevoice.md`, and temporary execution notes may be removed if
maintainers do not want them in the final change.

## 2. Baseline and branch safety

The original implementation must remain available for comparison and rollback:

| Purpose | Ref | Commit | Mutation policy |
| --- | --- | --- | --- |
| Known-working development baseline | `origin/feat/vibevoice-support` | `e0290cbc` | Do not rewrite or force-push |
| Full remediation work | `origin/fix/vibevoice-review-remediation` | Forked from `e0290cbc` | All reviewed fixes go here |
| Original review base | `67c54777` | Parent of `cadf0a60` | Read-only comparison point |

The owner reports that the preserved implementation passes stress testing on the
development machine. That is valuable rollback evidence, but it is not a
replacement for checked-in quality thresholds, repeatable benchmark commands,
or CI coverage.

Use these comparisons consistently:

```bash
# Original implementation review

git diff 67c54777...e0290cbc

# All remediation changes relative to the preserved implementation

git diff e0290cbc...HEAD
```

Rules:

1. Never commit remediation changes to `feat/vibevoice-support`.
2. Do not force-push the preserved source branch.
3. Use DCO sign-off for every new remediation commit.
4. Keep one logically reviewable module per commit or small commit series.
5. Do not rewrite remediation history until the final DCO/rebase phase unless
   explicitly approved.
6. A safety tag for `e0290cbc` may be added later, but only with owner approval.

## 3. Execution protocol

Every remediation module follows this protocol:

1. The assistant presents one detailed module plan, including files, interface
   changes, tests, risks, and acceptance criteria.
2. The owner reviews and explicitly approves or revises the plan.
3. Only then is that module implemented.
4. All available CPU/static tests are run on this machine.
5. GPU-only validation is recorded as `GPU_PENDING`; it is never reported as
   passed without a real GPU run.
6. The owner runs the recorded GPU commands on the development machine and
   returns the results.
7. This README is updated with results, decisions, and commit SHA before the next
   module begins.

Allowed status values:

- `NOT_STARTED`
- `PLANNED`
- `APPROVED`
- `IMPLEMENTING`
- `STATIC_VERIFIED`
- `DEV_MACHINE_TEST_PENDING`
- `CPU_VERIFIED`
- `GPU_PENDING`
- `GPU_VERIFIED`
- `BLOCKED`
- `DONE`

## 4. Environment constraints

The current machine has no GPU. The currently discovered Python environments
also do not provide a complete runnable stack containing all of `torch`, `vllm`,
`transformers`, `diffusers`, `soundfile`, and `pytest` together.

Therefore:

- maximize pure CPU unit, protocol, parser, config, state-machine, and static
  coverage;
- use dependency injection/fakes only for behavior that is genuinely independent
  of CUDA;
- do not use mocks to claim CUDA graph, numerical parity, VRAM, RTF, or
  concurrency correctness;
- keep GPU tests deterministic and ready to run on the development machine;
- record exact GPU commands, hardware, dependency versions, and outputs;
- do not install or upgrade heavyweight dependencies without explicit approval.

The owner elected to run all pytest and GPU suites later on the development
machine rather than continue installing packages on this macOS checkout. The
maintained launch commands and result template are in [TESTING.md](TESTING.md).

## 5. Review scope and summary

The reviewed range contains:

- 31 commits;
- 67 changed files;
- approximately 11,701 added lines;
- approximately 5,132 added production Python lines;
- approximately 995 added lines in shared runner/frontend code.

Overall result: the functional implementation is promising and the request-owned
state design has several good foundations, but the branch is not yet ready for
community merge. The primary blockers are:

1. DCO and static contribution gates;
2. request cleanup correctness;
3. GPU hot-loop synchronization and unbounded CUDA graph caching;
4. raw/browser streaming behavior that contradicts documentation;
5. missing real E2E, speaker-conditioning, quality, TP=2, and CI gates;
6. model-specific branches in shared TTS serving code;
7. a large shared runner extension without a linked RFC;
8. bundled voice files without recorded provenance and redistribution rights.

## 6. Standards findings

### ST-01 — DCO failure (`P0`)

All 31 original feature commits lack `Signed-off-by`. At least two author
identities appear in the range. Before upstream submission, history must be
squashed or rewritten into logical commits and every resulting commit must have
a matching DCO trailer.

### ST-02 — Ruff version mismatch (`RESOLVED`)

The original review used the locally installed Ruff 0.11.0 and reported 16
`UP038` findings. M1 reran the complete 56-file range with the repository-locked
Ruff 0.14.10; both `ruff check` and `ruff format --check` passed without those
changes. The earlier finding was a tool-version false positive, so no unsafe or
unnecessary `isinstance` rewrites were applied.

### ST-03 — Test mark failure (`RESOLVED IN M1`)

`tests/dfx/test_stability_helpers.py` had neither a CI level mark nor a hardware
mark. M1 added explicit `core_model` and `cpu` marks; the repository checker now
passes. A path move is deferred unless the later CI design requires one.

### ST-04 — SPDX and attribution (`RESOLVED IN M1`)

Among 35 added Python files:

- none uses the required vLLM-Omni SPDX copyright text exactly;
- `tests/model_executor/models/vibevoice/test_vibevoice_processing.py` has no
  SPDX license line;
- files adapted from Microsoft/Transformers must preserve appropriate upstream
  attribution in addition to vLLM-Omni attribution.

Target form for original vLLM-Omni code:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
```

M1 fixed all 35 added Python files and the additional staged files checked by
the current upstream SPDX hook. The config shim also retains explicit
Microsoft/Transformers attribution.

### ST-05 — Forbidden imports (`RESOLVED IN M1`)

The reviewed range introduced stdlib `base64`, stdlib `re`, and a direct
`hf_hub_download` call. M1 changed the newly introduced call sites to
`pybase64`, `regex`, and vLLM's `get_hf_file_to_dict` repository helper. The
upstream-derived `huggingface_hub.dataclasses.strict` decorator is not a Hub
repository/network API and is not forbidden by the current upstream checker; it
is retained for Transformers config validation. The current upstream forbidden
import checker passes on all M1-staged Python files.

### ST-06 — Direct CUDA usage (`PARTIALLY RESOLVED; PLATFORM REVIEW PENDING`)

The named-KV free-memory preflight now uses
`current_omni_platform.get_free_memory()`. Remaining direct CUDA calls are
limited to CUDA graph, stream/event, pinned-copy, and diagnostics paths in the
VibeVoice model package. They remain NVIDIA-specific and require either an
approved platform abstraction or an explicit CUDA-only maintainer decision
before final submission.

### ST-07 — TTS Adapter migration violation (`RESOLVED; TEST_PENDING`)

Request finalization, sampling overrides, finish headers, stage resolution, and
WebSocket response-format policy now flow through generic Adapter/OutputPolicy
hooks. Both `serving_speech.py` and `serving_speech_stream.py` contain no
executable VibeVoice model-name branch. CPU and WebSocket tests await execution
on the development machine.

### ST-08 — Missing RFC and oversized scope (`PARTIALLY RESOLVED`)

The change adds more than 5K lines of production code and a substantial shared
runner extension, especially:

- `vllm_omni/worker/named_kv_branch.py`
- `vllm_omni/worker/gpu_model_runner.py`
- `vllm_omni/worker/gpu_ar_model_runner.py`
- `vllm_omni/entrypoints/openai/serving_speech.py`

The named-KV ownership/lifecycle contract is now tracked in
[`docs/design/named_kv_branch.md`](../named_kv_branch.md), and the Adapter seam
references issue #4327. The original scope remains large, and the final named-KV
submission still needs a linked upstream issue plus logical PR/commit splitting.

### ST-09 — Typing and module depth (`PARTIALLY RESOLVED; P1`)

Graph replay now narrows captured Optional buffers through explicit runtime
invariants, semantic latent is Tensor-valued on eager and graph paths, and the
critical runner/model metadata keys have one canonical definition. A targeted
mypy run is still pending and dynamic upstream config/cache interfaces retain
intentional `Any` usage.

`vibevoice.py` is over 1,100 lines and mixes weight mapping, model
composition, multimodal encoding, lifecycle integration, output construction,
and graph dispatch. Internal labels such as `M4a`, `M4b`, `C1`, and `C2` depend
on untracked design context. The implementation should be reorganized into deep
modules with small lifecycle interfaces and tracked design terminology.

## 7. Contract audit: I1–I5

| ID | Requirement | Status | Review result |
| --- | --- | --- | --- |
| I1 | Explicit Delta/Cumulative streaming contract | Implemented; test pending | M2 documents and marks VibeVoice output as Delta, with model/router/consumer regression coverage awaiting development-machine pytest. |
| I2 | Robust output consumer types | Implemented; test pending | M2 adds explicit `None` handling, marker-aware list semantics, scalar/array normalization, and dtype/shape/sample-rate assertions. |
| I3 | No GPU synchronization in hot loop | Implemented; profiler pending | M5A removed per-token `.item()`/D2H token inspection; clone and disabled-instrumentation optimization remains evidence-gated. |
| I4 | Complete validation pyramid | Implemented; execution/thresholds pending | CPU/GPU/E2E, independent completion, fail-closed SSE, browser, merge CI, quality, performance, and stability procedures are checked in but not yet executed. |
| I5 | Request-owned state keyed by reserved request ID | Implemented; test pending | M2 uses `_omni_req_id`, cleans unscheduled finishes before early return, synchronizes pending D2H copies, and clears model state before named-KV shutdown. |

### I1 — Delta streaming contract

The current chain generally behaves as Delta:

1. stateful decoding produces one waveform chunk per audio transition;
2. chunks are drained once;
3. sparse routing yields one per-request Tensor;
4. final output uses `CONCAT_LAST`.

Required remediation:

- state the Delta contract in the model `forward()`/output interface;
- define behavior for empty steps and sample-rate metadata;
- test streaming emissions and final consolidation through the real output
  processor;
- stop inferring cumulative-vs-delta semantics merely from whether a value is a
  Python list.

### I2 — Output consumer hygiene

Required consumer inputs are `None`, Tensor, list of Tensors, NumPy ndarray, and
scalar forms where applicable. Tests must explicitly assert:

- source and normalized dtype;
- one-dimensional mono shape;
- 24,000 Hz sample rate in scalar and list metadata forms;
- finite, non-empty waveform;
- expected duration/chunk length.

Do not use truthiness fallback on Tensor/array values. Prefer explicit `is None`
checks.

### I3 — GPU hot loop

M5A now supplies control-token IDs from the scheduler-maintained CPU mirror and
the VibeVoice model package contains no per-token `.item()` fallback. A GPU
profiler trace must still confirm that no replacement synchronization was
introduced.

The negative path also clones an entire hidden-state batch and later clones
per-request conditions again. Ownership should be transferred or consumed in a
single transition with at most one required copy.

The production hot loops no longer carry disabled timing context managers or
other model-local instrumentation overhead.

### I4 — Validation pyramid

The branch needs all of the following before community merge:

- single request output contract;
- independently finishing batched requests;
- incremental raw and SSE streaming;
- reference-speaker conditioning quality;
- browser playback;
- official-weight/golden quality validation;
- 4+ concurrent request isolation;
- repeatable RTF/TTFP/VRAM benchmarks;
- CI registration and long-running stability.

VibeVoice is a single-stage model, so `async_chunk=false` is correct and should
not be changed merely to satisfy a multi-stage setting. Its equivalent contract
is `RequestOutputKind.DELTA` with incremental raw/SSE output.

### I5 — Request state ownership and cleanup

Most side state is keyed by the internal request ID, but request metadata should
use the reserved `_omni_req_id` field to avoid collision with user-provided
metadata.

Current cleanup is deferred. If a finish callback is followed by a runner call
with zero scheduled tokens, the runner returns before flushing deferred cleanup.
State, convolution caches, pinned host buffers, and named-KV blocks can remain
until another request arrives or the process exits.

Required behavior:

- immediately clean `finished_ids - scheduled_ids`;
- defer only `finished_ids & scheduled_ids` until final postprocessing;
- flush eligible cleanup before every early return;
- clear model-owned request state before closing the named-KV branch at shutdown;
- safely retire or synchronize pending pinned D2H transfers on abort;
- verify state and block counts return to zero under finish, abort, failure, and
  shutdown paths.

## 8. Additional functional findings

### FN-01 — Raw/browser streaming contradiction (`RESOLVED IN M2`)

M2 removes the VibeVoice raw-stream rejection and adds adapter plus official-
weight raw PCM coverage. Raw transport intentionally has no structured terminal
`finish_reason`; clients requiring it use SSE. Browser/AudioWorklet validation
remains a separate M6 gate.

### FN-02 — Unbounded request-driven CUDA graph cache (`IMPLEMENTED IN M3A; TEST_PENDING`)

M3A limits request controls before engine execution and makes the graph cache a
finite deployment policy:

- only `guidance_scale` and `num_diffusion_steps` are accepted;
- guidance must be a finite real number in `[0.0, 20.0]`;
- steps must be an integer in `[1, 50]`, without bool/string/float coercion;
- only official controls `(guidance=1.3, steps=10)` and batches 1 through 4
  create graph entries;
- all other valid controls and larger batches return to eager without growing
  or disabling the official graph cache;
- malformed and unknown deployment runtime config now fails startup;
- limits, eager fallback, and development-machine GPU gates are documented.

Static verification passes. CPU/GPU pytest and graph VRAM measurements remain
`DEV_MACHINE_TEST_PENDING`.

### FN-03 — Config schema compatibility (`DEFERRED TO FINAL REBASE`)

The current Microsoft/early-flat checkpoint path remains unchanged. A full
canonical nested-schema rewrite is not required for the known development
configuration and has a comparatively large model-startup/serialization blast
radius. Per owner decision, this finding remains open until the final upstream
base and its actual Transformers VibeVoice registration are locked. The final
submission must then adapt the real upstream class, narrow the support claim, or
constrain the dependency; it must not silently claim all three schemas.

### FN-04 — Decode graph return-type mismatch (`IMPLEMENTED IN M3C; TEST_PENDING`)

M3C captures and replays the Tensor-valued semantic latent rather than returning
`None`. GPU tests compare audio, semantic latent, and next embedding against
eager execution over consecutive tokens and segment reset. CUDA execution
remains `DEV_MACHINE_TEST_PENDING`.

### FN-05 — Default voice asset provenance (`AUDITED; BLOCKED`)

The tracked [asset provenance audit](ASSET_PROVENANCE.md) records hashes,
technical metadata, repository history, immutable first-party source searches,
and the evidence required for redistribution. No source, asset license, creator,
voice identity/consent, or redistribution permission was established. The files
remain only for preserved-branch validation and must be removed from the final
submission unless sufficient primary evidence and maintainer/legal approval are
recorded.

## 9. Test coverage review

### Existing useful coverage

- online WAV/PCM smoke checks;
- offline four-request generation;
- SSE length-cap aggregate-length check;
- multi-speaker generation and uploaded voice lifecycle;
- CPU state-machine/config/processing/weight tests;
- GPU diffusion/decode/named-KV tests;
- one-hour stability configuration with max concurrency 4.

### Missing or insufficient coverage

#### TEST-01 — Single request contract

Online coverage is useful, but offline tests must also assert output source and
normalized dtype, shape, `sr`, finite samples, and duration.

#### TEST-02 — Independent batch completion (`IMPLEMENTED; GPU_PENDING`)

The online suite now issues four concurrent requests capped at 2, 3, 4, and 5
tokens across official/custom controls and checks each response against its own
exact sample count. Client abort is followed by a bounded probe; mixed batch
success/failure, schedule-aware cleanup, and four-request offline generation are
also covered. Development-machine execution must still establish the result.

#### TEST-03 — Incremental streaming (`IMPLEMENTED; GPU_PENDING`)

The SSE E2E requires exactly two delta events of 3,200 samples followed by one
done event for a two-token length cap. Benchmark and browser consumers now fail
closed on `speech.audio.error`, missing done, duplicate done, or audio after
done. For both raw PCM and SSE, development-machine validation must:

- receive at least two non-empty chunks before completion;
- assert each transition contributes exactly 3,200 samples;
- assert no prefix replay or truncation;
- verify aggregate duration and sample rate;
- record first-chunk arrival separately from completion.

#### TEST-04 — Speaker conditioning

For identical text with different references, use a speaker embedding model and
assert a measured self-reference similarity margin. Merely producing non-empty
or byte-different audio is insufficient.

#### TEST-05 — Official quality gate

Restore an official Transformers/Microsoft reference or equivalent golden test.
Seed-TTS WER/CER/SIM thresholds must be non-optional in the gate. Establish
thresholds from repeated baseline runs and record the baseline environment.

#### TEST-06 — TP=2 (`CLAIM WITHDRAWN; OPTIONAL EXPERIMENT PENDING`)

User documentation now labels TP=2 experimental and unverified. It must not be
restored as a support claim without rank-consistent RNG, waveform, quality,
latency, and per-rank memory evidence from a real TP=2 generation gate.

#### TEST-07 — CI registration (`MERGE REGISTERED; CI_PENDING`)

The merge pipeline now has a source-gated one-H100 VibeVoice job covering the
official-weight offline and online suites under the strict both-graph overlay;
eligible capture failure cannot silently become eager. Source dependencies cover
config, protocol, output, Adapter, runner, model, and test paths. Ready's
existing core CUDA model-executor job selects the VibeVoice GPU unit tests.
Actual Buildkite execution is pending; nightly quality/performance and long
stability wiring remains evidence-gated rather than claiming unestablished
thresholds.

#### TEST-08 — Browser playback (`IMPLEMENTED; MANUAL_PENDING`)

`examples/online_serving/text_to_speech/vibevoice/` provides a same-origin
Gradio proxy, fail-closed raw/SSE player, phase-preserving 24 kHz resampling,
AudioWorklet backpressure, abort, metrics, and a four-tab isolation checklist.
Browser execution and saved artifacts remain a manual development-machine gate.

## 10. Remediation modules

The preferred upstream strategy is a small PR stack. If the work remains in one
branch, the same module separation still applies to commits and review.

| Module | Scope | Current status | Main acceptance gate |
| --- | --- | --- | --- |
| M0 | Baseline preservation and remediation tracker | `DONE` | Both remote branches preserve `e0290cbc`; this README is tracked |
| M1 | Contribution hygiene and CPU test baseline | `DEV_MACHINE_TEST_PENDING` | Static gates pass; pytest awaits the development machine |
| M2 | Streaming output and request lifecycle correctness | `DEV_MACHINE_TEST_PENDING` | Runtime contracts and tests are implemented; pytest/GPU execution awaits the development machine |
| M3 | Request controls, config compatibility, and graph safety | `M3A/M3C TEST_PENDING; M3B DEFERRED` | Controls/cache and semantic output implemented; schema decision waits for final rebase |
| M4 | TTS Adapter and runner module seams/RFC | `DEV_MACHINE_TEST_PENDING` | VibeVoice shared-serving branches removed; named-KV lifecycle documented/tested |
| M5 | GPU hot-path and CUDA graph optimization | `M5A TEST_PENDING; M5B EVIDENCE_PENDING` | Per-token control-ID D2H sync removed; further clone/graph changes require measurements |
| M6 | Full validation pyramid and CI | `CI/GPU/MANUAL_PENDING` | Abort/mixed-control E2E, H100 merge job, and browser client are checked in |
| M7 | User docs, assets, submission history, and final rebase | `ASSET_BLOCKED; REBASE_PENDING` | Provenance audit found no redistribution evidence; final history/schema decisions remain |
| M8 | Pre-development validation hardening | `STATIC_VERIFIED; DEV_MACHINE_TEST_PENDING` | Fail-closed consumers, strict graph evidence, independent completion, docs/example/CI alignment, and tonight's runbook are checked in |

### M0 — Baseline preservation and tracker

- preserve `origin/feat/vibevoice-support` at `e0290cbc`;
- branch `fix/vibevoice-review-remediation` from exactly that commit;
- create and maintain this README;
- do not modify production code.

### M1 — Contribution hygiene and CPU baseline

Implemented in `6e495feb`:

- corrected SPDX headers and preserved upstream attribution;
- verified that locked Ruff 0.14.10 has no `UP038` failures;
- replaced newly introduced forbidden imports through approved abstractions;
- repaired stability-helper level/hardware marks;
- added remote/local/error contract tests for tokenizer metadata resolution;
- added low-risk processing/config type annotations while deferring the
  processor return-interface mismatch to M4;
- added [TESTING.md](TESTING.md) as the development-machine launch runbook.

Static checks pass. Per owner direction, pytest execution is recorded as
`DEV_MACHINE_TEST_PENDING`; no CPU/GPU runtime pass is claimed from this
machine.

### M2 — Streaming output and request lifecycle

Implemented in `3b9a42aa` with runtime changes limited to the approved seams:

- formalized and marked the VibeVoice Delta output interface;
- made streaming consumption marker-aware while preserving unmarked legacy
  cumulative-list behavior;
- normalized scalar/Tensor/array chunks and sample-rate metadata explicitly;
- restored raw PCM streaming while retaining SSE finish metadata;
- migrated VibeVoice state lookup to runner-owned `_omni_req_id`;
- made finish cleanup aware of requests that really execute the current step;
- synchronized pending waveform D2H events on cleanup;
- cleared model request state before named-KV branch shutdown;
- added CPU contract tests, official-weight raw E2E coverage, and a real CUDA
  pending-copy cleanup test.

Static gates pass. Per owner direction, all pytest and GPU execution remains
`DEV_MACHINE_TEST_PENDING` and is recorded in [TESTING.md](TESTING.md).

### M3 — Controls, config, and graph safety

M3A was implemented in `6df04667` with changes limited to request policy,
deployment parsing, and diffusion graph admission:

- whitelists the two supported request controls and rejects unknown keys;
- applies approved guidance `[0.0, 20.0]` and integer-step `[1, 50]` bounds;
- validates a multi-control update before mutating or creating request state;
- centralizes official generation defaults at guidance 1.3 and 10 steps;
- restricts diffusion CUDA Graphs to those defaults and batches 1 through 4;
- routes custom controls and larger batches to eager without cache growth;
- makes malformed or unknown `vibevoice_runtime_config` fail startup;
- adds CPU policy/adapter/state/config tests, GPU graph-bound tests, online 4xx
  coverage, user documentation, and development-machine commands.

Static gates pass; pytest, CUDA parity, and graph VRAM measurements remain
`DEV_MACHINE_TEST_PENDING`.

M3C was implemented in `3fe57bbe`:

- decode graph entries now retain and replay `semantic_latent`;
- the output interface remains Tensor-valued rather than becoming Optional;
- GPU parity covers audio, semantic latent, next embedding, consecutive tokens,
  and segment reset.

M3B is explicitly `DEFERRED TO FINAL REBASE` by owner decision.

### M4 — Adapter and runner seams/RFC

Implemented in `49a61de7`, `034082c7`, and `17df5fde`:

- request-ID finalization and sampling overrides are default Adapter hooks;
- VibeVoice owns max-token enforcement and finish-header output policy;
- stage-key resolution is registry-owned and rejects ambiguous declarations;
- all VibeVoice executable branches were removed from `serving_speech.py`;
- legacy adapters retain identity defaults and existing dispatch priority;
- [Runner-Owned Named Causal KV Branches](../named_kv_branch.md) records
  ownership, capacity, request identity, finish/abort/failure/shutdown ordering,
  limitations, and relation to AR-Diffusion KV;
- tests cover transactional bind failure and continued teardown after cleanup
  exceptions.

No new typed runner Protocol was introduced: only one model currently needs the
schedule-aware hook, while signature-based compatibility already supports the
legacy hook family. This avoids establishing a shallow one-adapter seam.

Preferred eventual PR split remains:

1. `[Core] Add request-owned named causal KV branch`;
2. `[Frontend] Add generic speech Adapter lifecycle hooks`;
3. `[Model] Add VibeVoice-1.5B TTS support`.

### M5 — GPU hot path and performance

M5A was implemented in `6d4b1c12`:

- VibeVoice declares that it needs scheduled input token IDs from the CPU batch;
- the runner injects a request-aligned reserved CPU tuple from its existing
  `token_ids_cpu` mirror;
- prefill/decode control transitions use that tuple and reject missing or
  misaligned metadata;
- VibeVoice model code no longer contains `.item()` and adds no replacement
  `.cpu()` or D2H copy.

M5B remains evidence-gated. The duplicate negative-condition clone, timing
instrumentation, manual graph implementation, and graph defaults are unchanged.
The runbook now separates a true full-eager baseline from positive-graph-only,
positive+diffusion, positive+decode, and positive+both measurements at
concurrency 1/2/4 before any of those runtime changes.

### M6 — Validation and CI

Implemented scaffolding in `4c7b887b`, `cc42c534`, `e483fa8c`, and
`aa436be0`:

- official online E2E adds four concurrent default/custom-control requests with
  independent 2/3/4/5-token completion contracts;
- SSE consumers fail closed and assert exact delta/done ordering;
- client abort and the one-hour stability loop end with bounded health probes;
- the merge pipeline registers official offline+online suites on one H100 with
  focused source dependencies;
- ready's existing core CUDA job covers model-executor GPU tests;
- the TTS-hub browser example supports raw PCM, fail-closed SSE, continuous
  resampling, worklet backpressure, abort, underrun/overflow counters, and
  four-tab isolation instructions.

All runtime, Buildkite, and browser results remain pending.

Required validation levels:

| Level | Required tests |
| --- | --- |
| L1 CPU | Output type matrix, Delta consolidation, control validation, config schemas, zero-step cleanup, adapter hooks |
| L2 GPU | Single waveform contract, raw/SSE streaming, four independently finishing requests, abort cleanup |
| L3 H100 | Official model generation, speaker conditioning, quality thresholds, TP=2 if documented |
| L4 performance | RTF, TTFP, E2E, P95, peak VRAM, cold/warm graph behavior at concurrency 1/2/4 |
| L5 stability | Long-running max concurrency 4+, no errors, memory growth, state leaks, or speaker/output crossing |

All tests must receive correct level and hardware markers and be wired into the
appropriate Buildkite pipeline.

### M7 — Docs, assets, and submission

M7A provenance research was recorded in `b2d43567`. The audit found no primary
evidence for source, licensing, redistribution, creator, or voice consent. User
docs now identify the block and recommend explicit references. The binaries are
not removed in this batch, but the final upstream submission must remove them
unless the evidence gate is satisfied.

Remaining scope:

- optionally validate TP=2 before restoring any support claim;
- resolve the M3B schema against the locked final upstream base;
- publish reproducible quality/performance results and approved thresholds;
- build docs with strict warnings;
- create a new submission branch rather than force-pushing preserved branches;
- reconstruct logical DCO-signed commits and run all final gates.

### M8 — Pre-development validation hardening

Implemented before the development-machine handoff:

- moved WebSocket format policy into generic `OutputPolicy` and removed the
  final shared-serving VibeVoice model-name branch;
- made benchmark and browser SSE consumers fail on server error or incomplete
  terminal protocol;
- added continuous browser resampling, worklet backpressure, and proxy-failure
  cleanup;
- rejected fractional/infinite runtime capacities, centralized critical runner
  metadata keys, preserved bind errors during transactional cleanup, and routed
  named-KV memory queries through the platform interface;
- added strict graph-capture mode and capture-success diagnostics for evidence-
  producing runs;
- strengthened SSE event and independent 2/3/4/5-token concurrency tests;
- made the stability suite issue final health and exact-sample probes;
- aligned the example hub, contributor guide, generated docs source, CI source
  dependencies, TP=2 wording, and development-machine runbook.

All pytest, GPU, Buildkite, browser, quality, performance, and stability results
remain pending on the development machine. Static checks performed on the
current machine are recorded in the verification log.

Recommended model PR title:

```text
[Model] Add Microsoft VibeVoice-1.5B TTS support
```

## 11. Acceptance commands

The complete development-machine launch sequence and result template are in
[TESTING.md](TESTING.md).

The exact list may evolve with the branch, but every final run should include the
following classes of checks.

### Static and CPU checks

```bash
git diff --check
ruff format --check <changed-python-files>
ruff check <changed-python-files>
python -m compileall -q vllm_omni tests
python tools/pre_commit/check_test_marks.py <changed-test-files>
pre-commit run --from-ref <locked-upstream-base> --to-ref HEAD
```

Run targeted pytest suites after each module, then all VibeVoice CPU suites. Do
not hide missing dependencies or skipped tests; record collected, passed,
failed, skipped, and deselected counts.

### GPU commands to run on the development machine

At minimum:

```bash
pytest -s -v tests/model_executor/models/vibevoice \
  -m 'core_model and cuda' --run-level core_model

pytest -s -v \
  tests/e2e/offline_inference/test_vibevoice_tts.py \
  tests/e2e/online_serving/test_vibevoice_tts.py \
  -m 'advanced_model and cuda' --run-level advanced_model

pytest -s -v tests/e2e/accuracy/vibevoice/test_vibevoice_quality.py

pytest -s -v tests/dfx/stability/scripts/test_stability_vibevoice.py \
  --test-config-file tests/dfx/stability/tests/test_vibevoice.json
```

The final commands must include browser, concurrency, performance, and cleanup
tests, plus TP=2 only if its support claim is later restored.

### Benchmark report requirements

Record:

- GPU model/count and peak VRAM;
- CUDA, driver, PyTorch, vLLM, vLLM-Omni, Transformers, and Diffusers versions;
- exact model revision and commands;
- prompt/reference dataset and request controls;
- warmup protocol and at least three measured runs;
- mean plus range or standard deviation;
- E2E, audio TTFP, RTF, throughput, and P95;
- concurrency 1, 2, and 4;
- cold graph capture separately from warm replay;
- exact pytest summary and quality thresholds.

## 12. Decisions requiring owner approval

| Decision | Recommendation | Status |
| --- | --- | --- |
| Split Core/Frontend/Model/Perf work into a PR stack | Yes; highest chance of community review | `PENDING` |
| Keep the original branch immutable | Yes | `APPROVED` |
| Restore raw PCM streaming | Yes; SSE retains finish metadata | `IMPLEMENTED IN M2; TEST_PENDING` |
| Bound request controls and graph admission | Guidance `[0, 20]`, steps `[1, 50]`; only official controls use graphs | `IMPLEMENTED IN M3A; TEST_PENDING` |
| Implement full nested config normalization now | No; decide against the actual final upstream Transformers class | `DEFERRED TO FINAL REBASE` |
| Retain TP=2 support claim | No until a real TP=2 rank-consistency/generation gate passes | `CLAIM WITHDRAWN; EVIDENCE_PENDING` |
| Retain bundled default voices | Only after provenance/license/consent approval | `BLOCKED; AUDIT FOUND NO EVIDENCE` |
| Remove additional hidden-state clones now | No; require profiler/benchmark evidence | `EVIDENCE_PENDING` |
| Keep manual CUDA graph optimization in the first model PR | No; prefer eager correctness first and a measured follow-up | `EVIDENCE_PENDING` |
| Rebase remediation branch onto latest upstream immediately | No; validate this batch, then create a separate submission branch | `DEFERRED` |

## 13. Verification log

### Original review at `e0290cbc`

| Check | Result |
| --- | --- |
| `git diff --check` | Passed |
| `ruff format --check` | Passed |
| `ruff check` | Local Ruff 0.11.0 reported 16 findings; locked Ruff 0.14.10 later confirmed this was a false positive |
| Test mark check | Failed: one file |
| Python compileall | Passed |
| JSON/YAML parsing | Passed |
| DCO | Failed: 31/31 commits unsigned |
| Required SPDX copyright text | Failed: 0/35 added Python files |
| Targeted mypy smoke | Failed; contains definite new typing problems |
| Real pytest/GPU E2E | Not run in the review environment |
| MkDocs/markdownlint/full pre-commit | Not run in the review environment |

### M1 — Contribution hygiene and CPU baseline

- Plan approved: 2026-08-22
- Implementation commit: `6e495feb`
- Status: `DEV_MACHINE_TEST_PENDING`
- Static results:
    - `git diff --check` — passed;
    - Ruff 0.14.10 check/format over 56 original changed Python files — passed;
    - `python -m compileall -q vllm_omni tests` — passed;
    - test mark checker — passed;
    - current-upstream SPDX checker over staged Python files — passed;
    - current-upstream forbidden-import checker over staged Python files — passed;
    - Markdown fence/link structure check — passed.
- Runtime results:
    - pytest — not run per owner direction; commands recorded in `TESTING.md`;
    - GPU — not applicable to M1 changes and not run.
- Remaining gates:
    - run M1 and full VibeVoice CPU suites on the development machine;
    - run full pre-commit after the eventual upstream rebase;
    - direct CUDA and TTS Adapter checks remain intentionally assigned to M5/M4.

### M2 — Streaming output and request lifecycle

- Plan approved: 2026-08-22
- Implementation commit: `3b9a42aa`
- Status: `DEV_MACHINE_TEST_PENDING`
- Static results:
    - `git diff --check` — passed;
    - Ruff 0.14.10 check/format over all M2 Python files — passed;
    - `python -m compileall -q vllm_omni tests` — passed;
    - test mark checker — passed;
    - current-upstream SPDX, forbidden-import, and direct-`torch.cuda` checkers — passed;
    - Markdown fence/link structure check — passed.
- Runtime results:
    - pytest — not run per owner direction; M2 commands are in `TESTING.md`;
    - GPU — not run; raw/SSE E2E and pending-D2H cleanup tests are checked in.
- Compatibility:
    - old one-argument finished hooks remain supported;
    - unmarked cumulative audio lists retain their previous tail behavior;
    - `_omni_req_id` injection is capability-gated to VibeVoice;
    - no controls, graph cache, hot-loop, or Adapter-seam refactor was included.
- Remaining gates:
    - execute M1/M2 CPU suites on the development machine;
    - execute CUDA cleanup and official-weight raw/SSE E2E;
    - run the latest TTS Adapter ratchet after M4 and the upstream rebase.

### M3A — Request controls and graph-cache safety

- Plan approved: 2026-08-22
- Implementation commit: `6df04667`
- Status: `DEV_MACHINE_TEST_PENDING`
- Static results:
    - `git diff --check` — passed;
    - Ruff 0.14.10 check/format over all M3A Python files — passed;
    - `python -m compileall -q vllm_omni tests` — passed;
    - test mark checker — passed;
    - Markdown fence/link structure check — passed.
- Runtime results:
    - pytest — not run per owner direction; focused M3A CPU/GPU commands are in
    `TESTING.md`;
    - GPU — not run; finite-cache, custom-eager, post-custom replay, and bitwise
    parity tests are checked in.
- Compatibility:
    - official deployment controls and maximum concurrency remain unchanged;
    - custom controls remain supported but use eager diffusion;
    - model training scheduler configuration remains unchanged;
    - no config-schema, decode-output, hot-loop, Adapter-seam, or graph-kernel
    refactor was included.
- Remaining gates:
    - run M3A CPU tests on the development machine;
    - run the focused graph suite and record four-key/custom-probe VRAM;
    - run official-weight invalid-request E2E;
    - validate M3C decode outputs on CUDA;
    - resolve deferred M3B against the final upstream base.

### M3C–M7A approved remediation batch

- Plan approved: 2026-08-22
- Commits:
    - `3fe57bbe` — preserve Tensor-valued decode graph outputs;
    - `49a61de7`, `034082c7` — route VibeVoice through generic TTS Adapter hooks and preserve prepared output policy;
    - `17df5fde` — define/test the named-KV lifecycle contract;
    - `6d4b1c12` — remove control-token device synchronization;
    - `4c7b887b` — cover mixed controls and client abort;
    - `cc42c534` — register official H100 merge validation;
    - `e483fa8c`, `aa436be0` — add and harden the browser AudioWorklet client;
    - `b2d43567` — record the bundled-voice provenance audit;
    - `122baf1f` — complete SPDX metadata on modified Adapter files.
- Status: `DEV_MACHINE_TEST_PENDING | CI_PENDING | BROWSER_PENDING |
  ASSET_BLOCKED`
- Static results:
    - `git diff --check` — passed;
    - Ruff 0.14.10 check/format over all batch Python files — passed;
    - `python -m compileall -q vllm_omni tests examples` — passed;
    - test mark checker — passed;
    - changed-file SPDX/forbidden-import/direct-`torch.cuda` checks — passed;
    - Buildkite YAML parse — passed;
    - inline/worklet JavaScript syntax checks — passed;
    - Markdown fence/link structure check — passed.
- Runtime results:
    - pytest/GPU — not run per owner direction; focused commands are in
    `TESTING.md`;
    - Buildkite — job registered but not executed from this machine;
    - browser — implementation checked in, manual four-tab gate not run;
    - quality/performance/stability — no thresholds or pass claims added.
- Compatibility and scope controls:
    - non-VibeVoice adapters retain identity lifecycle hooks;
    - shared serving contains no executable VibeVoice model-name branch;
    - named-KV runner behavior is unchanged except for additional tests/docs;
    - M5 does not change clones, graph kernels, graph defaults, or timing;
    - M3B, TP=2, asset removal, thresholds, rebase, and history rewrite remain
    deferred.
- Remaining gates:
    - execute the CPU and CUDA batch commands on the development machine;
    - run the source-gated H100 Buildkite job;
    - complete raw/SSE browser evidence and four-request isolation;
    - collect eager/graph performance and one-hour stability evidence;
    - remove bundled defaults unless the asset evidence gate is resolved;
    - perform M3B/final submission work only after locking upstream.

### M8 — Pre-development validation hardening

- Approved through the pre-development review handoff: 2026-08-22
- Commits:
    - `7baecae3` — complete generic Adapter WebSocket policy;
    - `28b699ea` — fail closed in benchmark/browser streaming clients;
    - `3c903aad` — harden runtime metadata, graph, config, and cleanup contracts;
    - `6d4724bf` — sharpen official E2E, quality, stability, and H100 CI gates.
- Status: `STATIC_VERIFIED | DEV_MACHINE_TEST_PENDING | CI_PENDING |
  BROWSER_PENDING | ASSET_BLOCKED`
- Static results:
    - `git diff --check` — passed;
    - Ruff 0.14.10 check/format over 29 changed Python files — passed;
    - `python -m compileall -q vllm_omni tests examples` — passed;
    - test mark checker over 15 changed test files — passed;
    - Buildkite and five test-overlay YAML files parsed; both diff-aware and
    `--all` uploader renders include the strict H100 job;
    - inline player/worklet JavaScript syntax — passed;
    - chunked-versus-contiguous streaming resampler parity smoke — passed;
    - generated TTS example content, Markdown local links, shell-block syntax,
    and changed-file whitespace checks — passed.
- Type-check note:
    - targeted mypy still reports the repository's invalid comma-separated
    `python_version` plus existing untyped Torch-decorator/`Any` return errors;
    it is not recorded as passing.
- Runtime results:
    - pytest/GPU — not run on this dependency-incomplete machine;
    - Buildkite/browser/quality/performance/stability — not run;
    - exact environment, full-eager, graph-strict, five-cell performance, and
    one-hour stability commands are in `TESTING.md`.

### Module result template

Copy this block when completing a module:

```markdown
#### Mx — <name>

- Plan approved: YYYY-MM-DD
- Commit(s): `<sha>`
- Status: `STATIC_VERIFIED | DEV_MACHINE_TEST_PENDING | CPU_VERIFIED | GPU_PENDING | GPU_VERIFIED | DONE`
- Files changed:
  - ...
- CPU/static commands:
  - `<command>` — `<result and counts>`
- GPU commands:
  - `<command>` — `GPU_PENDING | <result and counts>`
- Behavior changes:
  - ...
- Remaining risks:
  - ...
```

## 14. Update history

| Date | Change |
| --- | --- |
| 2026-08-22 | Created the remediation branch and recorded the complete review, execution protocol, module plan, and CPU/GPU verification policy. |
| 2026-08-22 | Implemented M1 static hygiene, corrected the Ruff-version finding, added tokenizer resolver tests, and added the development-machine test runbook. |
| 2026-08-22 | Implemented M2 Delta/raw streaming and request lifecycle cleanup contracts; runtime tests remain pending on the development machine. |
| 2026-08-22 | Implemented M3A strict request controls and finite default-only diffusion graph admission; runtime tests remain pending on the development machine. |
| 2026-08-22 | Deferred M3B until the final upstream rebase and implemented M3C semantic-output parity. |
| 2026-08-22 | Removed VibeVoice branches from shared speech serving and documented the named-KV lifecycle contract. |
| 2026-08-22 | Removed control-token `.item()` synchronization while leaving clone/graph changes evidence-gated. |
| 2026-08-22 | Added abort/mixed-control E2E, H100 merge CI, and the browser AudioWorklet example; execution remains pending. |
| 2026-08-22 | Audited bundled voice assets and recorded an unresolved redistribution/consent block. |
| 2026-08-22 | Completed M8 pre-development hardening, strict evidence overlays, fail-closed clients, independent completion gates, and the development-machine handoff runbook. |
