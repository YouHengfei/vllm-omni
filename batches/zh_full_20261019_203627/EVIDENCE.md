# VibeVoice PR2 ZH batch: full-dataset online evaluation (2,020 items)

Run ID `vv-online-gpu4-20261019-203627`. Evidence for PR #9105 at source commit
`93166c21b084d7af6bb1b8d9a8ee80d8be3c51f1` (recorded as `inputs.source.head` in the
manifest, captured on a clean tracked tree).

## Start here

- [Manifest](results/manifest.json): `status=complete`, both jobs `contaminated=False`,
  script/checkpoint/dataset SHA256 digests, planned job list, environment and hardware
- [Online report](results/online/report.json) and [online log](results/online/zh_2020_online.log):
  non-streaming and streaming (PCM chunk transfer) modes, per-item WER/SIM/RTF/TTFP
- [HF reference report](results/hf/report.json), [generation log](results/hf/zh_2020_hf_gen.log),
  [evaluation log](results/hf/zh_2020_hf_eval.log)
- [Omni inventory](results/freeze_omni.txt), [HF inventory](results/freeze_hf.txt)
- GPU [start snapshot](results/gpu_start_snapshot.csv), [start compute apps](results/gpu_start_compute_apps.csv),
  [end snapshot](results/gpu_end_snapshot.csv),
  online end snapshots ([gpu](results/online/zh_2020_online_gpu_end.csv),
  [compute apps](results/online/zh_2020_online_compute_apps_end.csv)),
  HF end snapshots ([gpu](results/hf/zh_2020_hf_gpu_end.csv),
  [compute apps](results/hf/zh_2020_hf_compute_apps_end.csv))
- WAV SHA256 lists: [HF](results/hf/wav_sha256.txt), [online non-streaming](results/online/nonstreaming_wav_sha256.txt),
  [online streaming](results/online/streaming_wav_sha256.txt) (2,020 entries each)

## Configuration

Seed-TTS-Eval ZH `meta.lst`, `random_seed=0`, all 2,020 rows, single-speaker reference
cloning, one request at a time. vLLM-Omni online server on GPU4 (H100 80GB), BF16, eager,
FA3, seed 0, CFG 1.3, 10 diffusion steps, `max_new_tokens=512`. HF reference: Transformers
5.19, torch 2.13.0, serial `generate`, same checkpoint and dataset selection. WER uses
Paraformer (seed-tts-eval ZH protocol); SIM uses WavLM-base-plus embedding cosine
(proxy, not speaker verification).

## Results

| Configuration | WER (mean / median) | SIM (mean / median) | Evaluated | ASR failures | RTF | TTFP p50 / p99 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| Omni online non-streaming | 0.0275 / 0.0000 | 0.8723 / 0.8847 | 2018 | 2 | 0.320 | — |
| Omni online streaming (PCM) | 0.0310 / 0.0000 | 0.8708 / 0.8846 | 2020 | 0 | 0.315 | 57.1 / 62.7 |
| HF serial generate | 0.0281 / 0.0000 | 0.8722 / 0.8840 | 2019 | 1 | 0.437 | n/a |

0 request failures; all items stopped naturally. The 2 online non-streaming and 1 HF ASR
failures are evaluation-side (ASR noise), not generation failures; WER/SIM means exclude
those items. RTF = per-request wall time / generated audio duration; online timers include
HTTP overhead while the HF timer wraps serial `generate`, so the ~1.37x ratio is not a
matched-boundary speedup claim.

`SHA256SUMS` in this batch directory covers every file in the batch.
