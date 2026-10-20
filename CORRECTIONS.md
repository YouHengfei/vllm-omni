# Documentation correction: token cap and single-pass quality claims

This note corrects explanatory prose in the original
[evidence publication](https://github.com/YouHengfei/vllm-omni/tree/d3e7988f03d05c815b3af070975d45a13f9751f0)
for PR #9105. It addresses the recipe/evidence consistency review of PR head
`6f15d60ccbf8da2888515edc4da6c23680d5644c` on 2026-10-20.

The generation source remains `93166c21b084d7af6bb1b8d9a8ee80d8be3c51f1`.
No model execution or quality evaluation was rerun. Original manifests,
reports, logs, environment inventories, GPU snapshots, execution scripts and
WAV hash lists are unchanged. This is a documentation correction, not a
replacement manifest or a full re-audit of other claims in the bundle.

## 1. Measured generation used 2048 tokens, not 512

The previous recipe, PR description and three `EVIDENCE.md` files incorrectly
stated `max_new_tokens=512`. The archived run records and scripts consistently
specify **2048** for measured generation:

- [EN manifest](batches/en_full_20261019_180405/results/manifest.json) and
  [ZH manifest](batches/zh_full_20261019_203627/results/manifest.json):
  `inputs.parameters.max_tokens = 2048`.
- [EN online report](batches/en_full_20261019_180405/results/online/report.json)
  and [ZH online report](batches/zh_full_20261019_203627/results/online/report.json):
  top-level `max_tokens = 2048`, serialized from the parsed generation argument.
- [Archived runner](scripts/run_pinned_online.sh): both `run_online_job` and
  `run_hf_job` explicitly pass `--max-tokens 2048`.
- [Online driver](scripts/eval_online_seedtts.py): passes `args.max_tokens` to
  both measured modes; their HTTP payloads set `max_new_tokens` to that value.
- [HF generator](scripts/gen_hf_seedtts.py): passes the parsed value to
  `model.generate(max_new_tokens=max_tokens)`. HF `report.json` is an evaluation
  of saved WAVs, not a separate generation-argument record.

The four archived scripts' SHA256 digests match `inputs.scripts` in both
manifests. These records establish the documented executed setting; this audit
is not a new request-payload capture. There is no evidence here of a measured
512-token run or a change to the original 2048-token manifests.

Warmup is separate: the online runner passes `--warmup 3`, and the online
driver uses a 256-token cap for those three items in each mode. HF uses
`--warmup 1` with the 2048-token cap. Do not substitute online warmup's 256
for the measured 2048-token setting. The short curl example with 512 tokens
is also an example request, not the full-dataset evaluation configuration.

## 2. Single-pass quality observations do not establish equivalence

Earlier conclusions such as "within noise", "no regression" and
"streaming preserves quality" were stronger than the supplied evidence.
They are superseded by the following qualification, without changing any
WER/SIM/RTF/TTFP measurements:

- EN streaming mean WER is **0.0869** versus **0.0571** non-streaming:
  approximately **+2.98 percentage points**. Both median WERs are zero, but
  that does not establish equality of the means or overall quality.
- WER/SIM are single-pass observations. Statistical equivalence and absence
  of regression have not been established. The cause of the EN difference
  is unestablished; that difference alone does not establish a runtime regression.
- Noise attribution or a stronger equivalence claim would require controlled
  paired/repeated measurements with an explicit acceptance tolerance.
- The existing RTF timer-boundary caveat remains: the observed HF/online
  ratio is not a matched-boundary speedup claim.

## Preservation and citation

The correction is intended as a new child commit on `vibevoice-pr2-evidence`,
not an amendment or force replacement of `d3e7988f0`. Its old prose and
checksums remain accessible at that immutable commit. Only explanatory Markdown
and the checksum manifests covering that Markdown change in this correction.

Use the corrected publication's fixed commit URL when citing its explanation;
use the original fixed manifest/report URLs when tracing historical measurements.
The historical tests remain anchored to `93166c21b`, not a full-dataset rerun of
the latest PR head. Root and per-batch `SHA256SUMS` cover the corrected files
within their respective scopes, excluding each checksum manifest itself.
