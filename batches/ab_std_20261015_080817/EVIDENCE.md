# VibeVoice PR1: audited 2026-10-15 generation and CPU quality supplement

Evidence for PR #8909 at source commit `2e558341bcf8cc19392319ebbb39e0f6cd29987d`.
This batch adds startup provenance records and retained audio for both backends. It is **not**
a fully verified automatic pin/run/complete protocol; limitations and the launcher defect are below.
No generation or timing was rerun during this audit.

## Start here

- [Audited manifest](results/manifest.json): eight original outputs plus one quality supplement
- [Effective result summary](provenance/effective_results.json): machine-readable selection of original generation metrics and effective quality reports
- [Source and artifact audit](provenance/source_and_artifact_audit.json)
- [Pre-audit top-level manifest](provenance/top_level_manifest.before_audit.json): superseded metadata, preserved for transparency; do not use its incorrect supplement stop count or isolation claims
- [Omni generation/common evaluator](scripts/eval_snapshot_seedtts.py), [HF generator](scripts/gen_hf_seedtts.py)
- [Archival launcher](scripts/run_pinned.sh): contains known metadata bugs; **do not rerun unchanged**
- [Omni inventory](results/freeze_omni.txt), [HF inventory](results/freeze_hf.txt)

All original reports, logs, command notes, inventories and child manifests are copied without
content changes. Only the top-level manifest is corrected; its earlier version is archived above.
`SHA256SUMS` covers every other file in this batch, including the superseded records.

## Effective results

One full pass after one same-size warmup batch per configuration: seed 0, BF16, max tokens 2048,
CFG 1.3 and ten diffusion steps. EN has 1,088 items; ZH has 2,020.
All original generation runs report natural termination for every item.

| Configuration | WER | SIM | Quality evaluated | ASR failures | Natural stops | Generation RTF |
| --- | --- | --- | --- | --- | --- | --- |
| EN Omni B1 | 0.0501 | 0.8623 | 1088 | 0 | 1088/1088 | 0.3171 |
| EN HF B1 | 0.0599 | 0.8649 | 1088 | 0 | 1088/1088 | 0.4464 |
| EN Omni B4 | 0.0745 | 0.8618 | 1088 | 0 | 1088/1088 | 0.1161 |
| EN HF B4 | 0.0563 | 0.8631 | 1088 | 0 | 1088/1088 | 0.1630 |
| ZH Omni B1 | 0.0249 | 0.8742 | 2019 | 1 | 2020/2020 | 0.3153 |
| ZH HF B1 | 0.0263 | 0.8718 | 2020 | 0 | 2020/2020 | 0.4419 |
| ZH Omni B4 (CPU quality supplement) | 0.0263 | 0.8731 | 2020 | 0 | 2020/2020 | 0.1141 |
| ZH HF B4 | 0.0286 | 0.8720 | 2020 | 0 | 2020/2020 | 0.1651 |

ZH Omni B1 retains one `empty_asr`; its WER/SIM means exclude that item. Natural termination
and ASR success are separate counters. The ZH Omni B4 quality row replaces an incomplete
1,993-item evaluation; it does not replace generation timing, audio or stop counts.

HF/Omni RTF ratios are approximately 1.408/1.404 for EN B1/B4 and 1.402/1.447 for ZH B1/B4.
These are observed normalized generation ratios, not identical-output latency speedups. Generated
durations differ, the timer boundaries are not strictly identical, and continuous GPU isolation
has not been independently established. No statistical equivalence is claimed. EN Omni B4 WER
is about 1.81 percentage points higher than HF B4 in this batch.

## Report and log links

| Configuration | Original report | Generation log |
| --- | --- | --- |
| EN Omni B1 | [JSON](results/vv-pinned-gpu1-20261015-081007/en_omni_b1/report.json) | [log](results/vv-pinned-gpu1-20261015-081007/en_omni_b1.log) |
| EN Omni B4 | [JSON](results/vv-pinned-gpu1-20261015-081007/en_omni_b4/report.json) | [log](results/vv-pinned-gpu1-20261015-081007/en_omni_b4.log) |
| EN HF B1 | [JSON](results/vv-pinned-gpu1-20261015-081007/en_hf_b1/report.json) | [log](results/vv-pinned-gpu1-20261015-081007/en_hf_b1_gen.log) |
| EN HF B4 | [JSON](results/vv-pinned-gpu1-20261015-081007/en_hf_b4/report.json) | [log](results/vv-pinned-gpu1-20261015-081007/en_hf_b4_gen.log) |
| ZH Omni B1 | [JSON](results/vv-pinned-gpu2-20261015-081008/zh_omni_b1/report.json) | [log](results/vv-pinned-gpu2-20261015-081008/zh_omni_b1.log) |
| ZH Omni B4 | [JSON](results/vv-pinned-gpu2-20261015-081008/zh_omni_b4/report.json) | [log](results/vv-pinned-gpu2-20261015-081008/zh_omni_b4.log) |
| ZH HF B1 | [JSON](results/vv-pinned-gpu7-20261015-081009/zh_hf_b1/report.json) | [log](results/vv-pinned-gpu7-20261015-081009/zh_hf_b1_gen.log) |
| ZH HF B4 | [JSON](results/vv-pinned-gpu7-20261015-081009/zh_hf_b4/report.json) | [log](results/vv-pinned-gpu7-20261015-081009/zh_hf_b4_gen.log) |

Each group also contains its original `run.log`, `provenance.json`, `manifest.json` and inventories;
HF evaluation logs are named `*_eval.log`. Original absolute paths refer to the original machine.
Top-level audited report paths are relative to `results/` after relocation.

## CPU supplement: same saved audio, different evaluation device

- [Supplement report](results/zh_omni_b4_reeval/report.json)
- [Full evaluation log](results/zh_omni_b4_reeval/reeval.log)
- [Original command note](results/zh_omni_b4_reeval/eval_command.txt)
- [2,020 input WAV hashes](results/zh_omni_b4_reeval/wav_sha256.txt)

The original ZH Omni B4 evaluation encountered 27 `asr_exception` results during Paraformer
initialization after proxy/download failures. Generation audio was retained. Re-evaluating all
2,020 saved WAVs produced zero ASR/SIM failures, mean WER `0.02627653916821879` and mean
SIM `0.8731304187585812`. All 2,020 current WAV hashes match the supplied supplement list.
There is no generation-time WAV hash list, so this is not proof of uninterrupted historical custody.

The supplement uses `.venv-test` and the same evaluator code, but **CPU rather than CUDA** for
Paraformer and WavLM. Among the original 1,993 successful items, one WER value and 1,989 SIM
values changed; this is a whole replacement quality evaluation, not merely appending 27 rows.
The WER mean changes from 0.0264 to 0.0263 at four decimal places; SIM rounds to 0.8731 in both.
Do not attribute the entire difference solely to the missing items.

Original generation RTF remains `0.1141355566251767`; natural stops remain **2020/2020**, not
1993/2020. The latter number was the incomplete evaluation count, incorrectly copied into the
pre-audit supplement metadata. CPU re-evaluation supplies no new generation isolation evidence.

## Provenance and launcher limitations

The launcher records HEAD, tracked dirty-file count, three script hashes, checkpoint config hash,
EN/ZH metadata hashes, GPU UUID and dependency inventories before starting jobs. These records
agree across the three GPU groups and identify current PR HEAD with zero recorded tracked changes.
There is no end-of-run source/driver identity recheck. The source audit records the current evaluator
hash; this post-run calculation must not be represented as a launch-time measurement.

All three original child manifests have `runs: []` and `status: all succeeded`:
[GPU1](results/vv-pinned-gpu1-20261015-081007/manifest.json),
[GPU2](results/vv-pinned-gpu2-20261015-081008/manifest.json),
[GPU7](results/vv-pinned-gpu7-20261015-081009/manifest.json).
In the archival launcher, the output filename shifts the per-job Python argument parsing,
so `int(batch)` receives `en` or `zh` and no metadata file is produced. `all([])` then yields a
false success indication. HF generation and evaluation also lack separate recorded exit codes.
These are local evidence-recording defects, not demonstrated model-runtime defects. The original
empty manifests are intentionally preserved; the top-level summary is audited against actual
reports/logs rather than presented as successful automatic assembly.

The checkpoint revision `d4cd82eb4371bf3e3d088e1ea7fa4eefb60449bf` is confirmed by the operator
and previous current-cache checks. This launcher hashes only `config.json`, not weight shards.
It hashes dataset metadata, not reference audio; the immutable upstream dataset revision is unknown.
The evaluator resolves Paraformer `master` and HF aliases rather than enforcing immutable revisions.
Supplement logs resolve WavLM to `4c66d4806a428f2e922ccfa1a962776e232d487b`, but a recorded
revision/config hash does not by itself prove all evaluation weights were pinned at execution.

## Isolation, timing and repeatability

[Startup snapshot](results/gpu_startup_snapshot.csv) and [end snapshot](results/gpu_end_snapshot.csv)
show GPU1/2/7 at 4 MiB and 0% utilization at those instants. No continuous process/occupancy trace is
retained. Uninterrupted exclusivity and original `contaminated=false` fields are operator declarations,
not detector conclusions. Idle endpoints do not imply that the GPUs remained idle during generation.

Omni prompt-string rendering/reference reads precede the timer; `generate()` and output sorting are
inside it. HF `apply_chat_template`, device transfer, generation, synchronization, CPU extraction and
stop/duration bookkeeping are inside its timer. Both exclude startup, warmup, reference reads, WAV
writes and quality evaluation. The recorded single passes are not a repeated matched-boundary benchmark.

Compared with the 2026-10-14 batch, only four of eight original per-item evaluations are identical:
EN HF B1/B4, EN Omni B1, ZH Omni B1. EN Omni B4, ZH Omni B4 and ZH HF B1/B4 differ.
Cross-run generation/quality differences remain unexplained; bitwise reproduction is not claimed.

## Integrity and scope

From this batch directory:

```bash
sha256sum -c SHA256SUMS
```

No WAVs, checkpoint weights, dataset files or local PR description are published here. Both backends'
12,432 WAVs remain local; headers, IDs and durations were checked in the preceding audit. Reports
contain target text and ASR transcripts; logs retain local paths and GPU identifiers. This evidence
update does not modify the model PR, its recipe, PR body, comments or review status.
