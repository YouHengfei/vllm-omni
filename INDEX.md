# VibeVoice PR2 evidence index

Evidence published separately from model PR #9105 (VibeVoice-1.5B online serving and
multi-speaker prompts). Raw artifacts are retained unchanged; checksums are scoped per batch.

[Documentation correction](CORRECTIONS.md): the measured token cap was 2048, not 512;
quality equivalence/no-regression conclusions are not established by these single passes.
The original [publication](https://github.com/YouHengfei/vllm-omni/tree/d3e7988f03d05c815b3af070975d45a13f9751f0)
remains available. This correction changes explanatory prose and checksums only, not raw
run artifacts or metrics; it is not a new test run or a full re-audit of other claims.

| Batch | Entry point | Status |
| --- | --- | --- |
| 2026-10-19 EN full-dataset online + HF reference (1,088 items) | [EN batch evidence](batches/en_full_20261019_180405/EVIDENCE.md) | `status=complete`, both jobs `contaminated=False` |
| 2026-10-19 ZH full-dataset online + HF reference (2,020 items) | [ZH batch evidence](batches/zh_full_20261019_203627/EVIDENCE.md) | `status=complete`, both jobs `contaminated=False` |

The tested source is commit `93166c21b084d7af6bb1b8d9a8ee80d8be3c51f1` of the PR branch
`feat/vibevoice-pr2-survey` on `github.com/YouHengfei/vllm-omni`. Both batch manifests record
that commit as `inputs.source.head`, captured on a clean tracked tree. At the original
publication, the PR branch had advanced by documentation-only commits. It subsequently
received upload-validation fixes; these historical runs are not a full-dataset rerun of
the latest PR head. The tested commit remains reachable in branch history.

Execution scripts are published under [scripts/](scripts/); each batch manifest records their
SHA256 digests, checkpoint and dataset hashes, both environment freezes, GPU start/end
snapshots, and the planned job list. Generated audio is not stored in git (~9.9 GB); per-file
SHA256 lists under each batch's `results/` identify every generated WAV.

Root `INDEX.md` is the navigation entry point; use fixed Git commit URLs when citing these
materials. No published evidence update changes the model PR's source, recipe, PR body or
review status.
