# VibeVoice PR2 EN batch: full-dataset online evaluation (1,088 items)

Run ID `vv-online-gpu4-20261019-180405`. Evidence for PR #9105 at source commit
`93166c21b084d7af6bb1b8d9a8ee80d8be3c51f1` (recorded as `inputs.source.head` in the
manifest, captured on a clean tracked tree).

## Start here

- [Manifest](results/manifest.json): `status=complete`, both jobs `contaminated=False`,
  script/checkpoint/dataset SHA256 digests, planned job list, environment and hardware
- [Online report](results/online/report.json) and [online log](results/online/en_1088_online.log):
  non-streaming and streaming (PCM chunk transfer) modes, per-item WER/SIM/RTF/TTFP
- [HF reference report](results/hf/report.json), [generation log](results/hf/en_1088_hf_gen.log),
  [evaluation log](results/hf/en_1088_hf_eval.log)
- [Omni inventory](results/freeze_omni.txt), [HF inventory](results/freeze_hf.txt)
- GPU [start snapshot](results/gpu_start_snapshot.csv), [start compute apps](results/gpu_start_compute_apps.csv),
  online end snapshots ([gpu](results/online/en_1088_online_gpu_end.csv),
  [compute apps](results/online/en_1088_online_compute_apps_end.csv)),
  HF end snapshots ([gpu](results/hf/en_1088_hf_gpu_end.csv),
  [compute apps](results/hf/en_1088_hf_compute_apps_end.csv))
- WAV SHA256 lists: [HF](results/hf/wav_sha256.txt), [online non-streaming](results/online/nonstreaming_wav_sha256.txt),
  [online streaming](results/online/streaming_wav_sha256.txt) (1,088 entries each)

`results/zh_1088_online.killed_before_start.log` is a ZH attempt from this run window that was
killed before generation started (see manifest `note`); the completed ZH run is the separate
batch `zh_full_20261019_203627`. The log is preserved unchanged for transparency.

## Configuration

Seed-TTS-Eval EN `meta.lst`, `random_seed=0`, first 1,088 rows, single-speaker reference
cloning, one request at a time. vLLM-Omni online server on GPU4 (H100 80GB), BF16, eager,
FA3, seed 0, CFG 1.3, 10 diffusion steps, `max_new_tokens=512`. HF reference: Transformers
5.19, torch 2.13.0, serial `generate`, same checkpoint and dataset selection. WER uses
Whisper-large-v3; SIM uses WavLM-base-plus embedding cosine (proxy, not speaker verification).

## Results

| Configuration | WER (mean / median) | SIM (mean / median) | Evaluated | ASR failures | RTF | TTFP p50 / p99 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| Omni online non-streaming | 0.0571 / 0.0000 | 0.8617 / 0.8729 | 1088 | 0 | 0.310 | — |
| Omni online streaming (PCM) | 0.0869 / 0.0000 | 0.8623 / 0.8745 | 1088 | 0 | 0.306 | 55.1 / 62.5 |
| HF serial generate | 0.0599 / 0.0000 | 0.8649 / 0.8742 | 1088 | 0 | 0.433 | n/a |

0 request failures; all items stopped naturally. RTF = per-request wall time / generated
audio duration; online timers include HTTP overhead while the HF timer wraps serial
`generate`, so the ~1.4x ratio is not a matched-boundary speedup claim. Streaming WER mean
is higher than non-streaming while the median stays 0; single-pass noise, not a regression
claim in either direction.

`SHA256SUMS` in this batch directory covers every file in the batch.
