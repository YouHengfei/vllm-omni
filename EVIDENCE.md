# Historical VibeVoice PR1 evidence: 2026-10-14

Historical evidence for PR #8909, published separately in the fork `YouHengfei/vllm-omni`
on branch `vibevoice-pr1-evidence`. It does not contain results from the new pinned rerun.

## Evidence status

[Manifest](results/manifest.json) is **retrospectively reconstructed**, not captured at execution time.
Report/log contents and current file hashes were checked after the run. Runtime working-tree
cleanliness, unchanged Python drivers, original launch commands and uninterrupted GPU exclusivity
are **test-operator attestations**, not independently preserved machine records.

The historical results remain useful observations, but the bundle does not close every provenance
gap. In particular, the immutable upstream dataset revision, contemporaneous driver/environment
fingerprints and continuous GPU occupancy traces are unavailable. Current hashes must not be
represented as hashes captured during the original execution.

## Contents

- [Omni generation and common evaluation driver](scripts/eval_snapshot_seedtts.py)
- [HF generation driver](scripts/gen_hf_seedtts.py)
- [Current equivalent launch script](scripts/run_one.sh)
- [Manifest, commands, parameters, environment and hardware](results/manifest.json)
- [Omni dependency inventory](results/freeze_omni_venv-test.txt)
- [HF dependency inventory](results/freeze_hf_venv-test-hf.txt)
- [Current dataset input hashes](results/dataset_inputs.sha256.json): metadata and all 1,676 distinct reference WAVs
- [Current checkpoint file hashes](results/checkpoint_files.sha256.json): includes all three weight shards
- [Committed-source coverage](provenance/source_coverage.json)
- [Candidate run commit to PR-head diff](provenance/run_candidate_to_pr_head.patch)
- `SHA256SUMS`: checksums for all other files in this bundle

Original JSON reports, logs and dependency inventories are copied without content changes.
The corrected manifest distinguishes natural termination from ASR evaluation outcomes.
The HF inventory is a 72-package `pip list` table, **not** an installable requirements file.
Both dependency inventories were collected after the historical runs.

## Eight reports

EN contains 1,088 items; ZH contains 2,020. Each configuration used one same-size warmup
batch and one measured full pass, seed 0, BF16, max tokens 2048, CFG 1.3, ten diffusion steps.

| Run        | Report                                     | Generation log                    | Natural stops | ASR failures | WER/SIM evaluated |
| ---------- | ------------------------------------------ | --------------------------------- | ------------- | ------------ | ----------------- |
| EN Omni B1 | [JSON](results/en_omni/omni-b1/report.json) | [log](results/en_omni/omni-b1.log) | 1088/1088     | 0            | 1088              |
| EN Omni B4 | [JSON](results/en_omni/omni-b4/report.json) | [log](results/en_omni/omni-b4.log) | 1088/1088     | 0            | 1088              |
| EN HF B1   | [JSON](results/en_hf/hf-b1/report.json)     | [log](results/en_hf/hf-b1-gen.log) | 1088/1088     | 0            | 1088              |
| EN HF B4   | [JSON](results/en_hf/hf-b4/report.json)     | [log](results/en_hf/hf-b4-gen.log) | 1088/1088     | 0            | 1088              |
| ZH Omni B1 | [JSON](results/zh_omni/omni-b1/report.json) | [log](results/zh_omni/omni-b1.log) | 2020/2020     | 1            | 2019              |
| ZH Omni B4 | [JSON](results/zh_omni/omni-b4/report.json) | [log](results/zh_omni/omni-b4.log) | 2020/2020     | 1            | 2019              |
| ZH HF B1   | [JSON](results/zh_hf/hf-b1/report.json)     | [log](results/zh_hf/hf-b1-gen.log) | 2019/2020     | 0            | 2020              |
| ZH HF B4   | [JSON](results/zh_hf/hf-b4/report.json)     | [log](results/zh_hf/hf-b4-gen.log) | 2020/2020     | 0            | 2020              |

HF reports contain evaluation metrics; HF generation timing and stop counts are in the generation
logs. The corresponding `hf-b*-eval.log` files and each group's `run.log` are also included.
All report means were recomputed from per-item entries. IDs/order and available target texts
match the current dataset selection. HF WAV durations match generation logs at their printed precision.

Omni ZH B1 has `empty_asr` for `00014150-00000644`; B4 has it for `00004978-00000074`.
These are different items. Empty ASR alone does not establish whether generation or recognition
caused the outcome. HF ZH B1's non-natural termination is **not** an ASR failure.

## Source coverage

The operator identifies `0d3a6ffa53ea71dbecb8610c1c76126cf550eeea` as the starting source commit.
Logs place the overall historical window at 07:55:11–10:34:28 +0800 on 2026-10-14.
The final HF evaluation overlaps the 10:34:16 amend to `375fb7cfa`, which only changes test assertions.

| Transition                     | Committed changes                               |
| ------------------------------ | ----------------------------------------------- |
| `0d3a6ffa5` → `375fb7cfa` | Four length-cap E2E assertion lines             |
| `375fb7cfa` → `1aace0d07` | Recipe update                                   |
| `1aace0d07` → `2e558341b` | Four-commit split; identical complete file tree |

All four commits have the same `vllm_omni/` tree object:
`2b064986b3aac60dcd0c7ec7737c5d7ce8eed97f`.
This includes the generation runtime and common quality evaluator. The tree comparison does not
prove there were no transient uncommitted changes during the historical run. The length-cap
assertions added later are not themselves validated by this full-dataset run.

## Scripts and reproduction limits

Python copies match the supplied SHA256 values. The operator confirms historical identity, but no
run-time hash capture/backup is supplied. The HF driver's current mtime falls within the run window;
its historical identity therefore remains an attestation rather than a hash-backed observation.
The original shell runner was deleted. The operator reports that the supplied replacement only
adapts repository/script paths; exact equivalence cannot be independently diffed.

The scripts remain unmodified evidence copies. To use them in a source checkout, place them under
`local_tests/vibevoice_pr1/` as expected by `run_one.sh`, configure the two Python environments, and
adapt the machine-specific model, dataset, cache and output paths. The original reconstructed
launch commands are in the manifest. Absolute paths in original reports refer to the original
machine and may no longer exist after directory relocation.

Checkpoint files currently match cached HF revision
`d4cd82eb4371bf3e3d088e1ea7fa4eefb60449bf`, including shard ETag verification.
Dataset hashes identify current local inputs, but no immutable upstream dataset revision is known.
Evaluation model config hashes/revisions are post-run cache observations; Paraformer `master` is
mutable and its config hash alone does not pin its weights.

Timing is not strictly identical across backends: Omni prompt-string rendering is outside its timer,
while HF `apply_chat_template` is inside. Omni times generation plus output ordering; HF also includes
CPU waveform extraction and stop/duration bookkeeping. Both exclude reference reads, startup,
warmup, WAV writes and quality evaluation. RTF ratios are historical observations, not exact
matched-boundary speedups or statistical-equivalence evidence.

## Exclusions and verification

No generated WAVs, checkpoint weights, dataset contents, PR description  or new pinned-rerun
artifacts are included. The 6,216 HF WAVs remain local (all checked as nonempty mono 24 kHz FLOAT).
Omni generated WAVs were not retained by this launch configuration. Reports contain dataset target
text and ASR transcripts; local paths and GPU UUIDs remain in the historical logs/manifest.

From this bundle directory, verify integrity with:

```bash
sha256sum -c SHA256SUMS
```

The source repository and current model PR are unchanged by preparing this bundle.
