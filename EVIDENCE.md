# VibeVoice PR2 evidence: 2026-10-19 full-dataset online evaluation

Evidence for PR #9105, published separately in the fork `YouHengfei/vllm-omni` on branch
`vibevoice-pr2-evidence`. Two batches: Seed-TTS-Eval EN (1,088 items) and ZH (2,020 items),
each covering vLLM-Omni online serving (non-streaming + streaming) and an HF Transformers
serial reference on the same checkpoint, dataset selection and seed.

## Evidence status

See [the documentation correction](CORRECTIONS.md) for the measured 2048-token
limit and the scope of the single-pass quality observations. Original run
artifacts and numerical results are unchanged; no generation or evaluation was rerun.

Both batch manifests record `status=complete` and per-job `contaminated=False`, with
GPU start/end snapshots and environment freezes captured at run time. The manifests are
execution-time records, not retrospective reconstructions.

Remaining provenance limitations, stated honestly:

- Continuous GPU exclusivity during the runs is an operator attestation supported by
  start/end snapshots, not an uninterrupted machine record.
- WER/SIM are single-pass observations. Statistical equivalence and absence of
  regression have not been established. Differences cannot be attributed to noise
  without controlled paired/repeated evidence and an explicit acceptance tolerance.
- SIM uses WavLM-base-plus embedding cosine (a proxy), not the seed-tts-eval UniSpeech
  speaker-verification checkpoint.
- Generated audio is identified by SHA256 lists, not stored in this branch (~9.9 GB).

## Contents

- [EN batch](batches/en_full_20261019_180405/EVIDENCE.md): 1,088 items, 3 configurations
- [ZH batch](batches/zh_full_20261019_203627/EVIDENCE.md): 2,020 items, 3 configurations
- [scripts/](scripts/): the four execution drivers, byte-identical to the SHA256 digests
  recorded in each batch manifest (`eval_online_seedtts.py`, `gen_hf_seedtts.py`,
  `eval_snapshot_seedtts.py`, `run_pinned_online.sh`)
- Per batch: `results/manifest.json`, run/generation/evaluation logs, per-configuration
  `report.json`, dependency freezes, GPU snapshots, WAV SHA256 lists, per-batch `SHA256SUMS`.

All original reports, logs, inventories and manifests are copied without content changes.
`SHA256SUMS` in each batch covers every file in that batch.

## Result summary

Head `93166c21b`, H100 80GB, BF16, eager, FA3, seed 0, CFG 1.3, 10 diffusion steps,
`max_new_tokens=2048`, single-speaker reference cloning, B1 serial. HF reference runs
Transformers 5.19 in a separate environment. 0 request failures across ~6,200 generations.

| Locale | Backend | Transport | Items | WER (mean) | SIM (mean) | RTF | TTFP p50 / p99 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EN | Omni online | non-streaming | 1088 | 0.0571 | 0.8617 | 0.310 | — |
| EN | Omni online | streaming (PCM) | 1088 | 0.0869 | 0.8623 | 0.306 | 55 / 62 |
| EN | HF reference | serial generate | 1088 | 0.0599 | 0.8649 | 0.433 | n/a |
| ZH | Omni online | non-streaming | 2018 | 0.0275 | 0.8723 | 0.320 | — |
| ZH | Omni online | streaming (PCM) | 2020 | 0.0310 | 0.8708 | 0.315 | 57 / 63 |
| ZH | HF reference | serial generate | 2019 | 0.0281 | 0.8722 | 0.437 | n/a |

EN streaming mean WER is 0.0869 versus 0.0571 non-streaming (+2.98 percentage
points), despite both medians being zero. The cause is unestablished; this
difference alone does not establish a runtime regression.

Items column is the WER-evaluated count; ZH online non-streaming has 2 ASR failures and
ZH HF has 1 (ASR noise, not generation failure). RTF timer boundaries differ between
backends (online includes HTTP overhead; HF times serial `generate`), so the ~1.4x
ratio is an observed normalized ratio, not a matched-boundary speedup claim.
