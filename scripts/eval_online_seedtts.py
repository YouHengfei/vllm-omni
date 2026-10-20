# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online VibeVoice Seed-TTS-Eval WER/RTF/TTFP evidence (local, not part of the PR).

Drives the running vllm-omni /v1/audio/speech server with VibeVoice's request
shape (ref_audio inline data: URL, no task_type/ref_text/language — those are
rejected by the adapter). Measures per-item:

  * non-streaming WAV : wall (POST -> full body), audio seconds -> RTF
  * streaming   PCM   : TTFP (time to first audio chunk), wall, audio seconds -> RTF

WER/SIM are computed with the same framework harness as PR1
(compute_seed_tts_wer_metrics over SeedTTSSampleRequest + tts_output_pcm_bytes),
so the selection and eval path are identical to local_tests/vibevoice_pr1.

HF TTFP baseline: NOT measurable. HF generate() (transformers
VibeVoiceForConditionalGeneration._sample) only streams text tokens via its
streamer; audio chunks are accumulated and returned at the end
(generation_vibevoice.py:720/736/741). Use PR1's gen_hf_seedtts.py for the
HF WER/RTF baseline and compare those two metrics only.

Run under .venv-test on a free GPU, e.g.:

  CUDA_VISIBLE_DEVICES=4 .venv-test/bin/python local_tests/vibevoice_pr2/eval_online_seedtts.py \
      --model vibevoice/VibeVoice-1.5B-hf \
      --dataset /SharedData/youhf/dataset/Seed-TTS-Eval/en \
      --items 40 --warmup 3 --output local_tests/vibevoice_pr2/results/eval_online_en
"""

import argparse
import base64
import io
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

from tests.helpers.runtime import OmniServer
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.benchmarks.data_modules.seed_tts_dataset import SeedTTSDataset, SeedTTSSampleRequest
from vllm_omni.benchmarks.data_modules.seed_tts_eval import (
    compute_seed_tts_wer_metrics,
    print_seed_tts_wer_summary,
)


def load_items(locale_dir: Path, count: int, seed: int):
    ds = SeedTTSDataset(dataset_path=str(locale_dir.parent), random_seed=seed, locale=locale_dir.name)
    rows = ds.data
    if count > len(rows):
        raise SystemExit(f"requested {count} items but meta.lst has {len(rows)}")
    return [(r.utterance_id, locale_dir / r.prompt_wav_rel, r.target_text) for r in rows[:count]]


def ref_data_url(wav_path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(wav_path.read_bytes()).decode()


@dataclass
class _Out:
    success: bool = True
    error: str = ""
    tts_output_pcm_bytes: bytes = b""


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def eval_pcm(items, pcm_list, locale):
    requests_ = [
        SeedTTSSampleRequest(
            prompt=text,
            prompt_len=0,
            seed_tts_utterance_id=utt,
            seed_tts_locale=locale,
            seed_tts_ref_wav_path=str(wav),
        )
        for utt, wav, text in items
    ]
    return compute_seed_tts_wer_metrics(requests_, [_Out(tts_output_pcm_bytes=p) for p in pcm_list], include_per_item=True)


def session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False  # avoid conflicting proxy env vars
    return s


def run_nonstreaming(url, model, items, max_new_tokens):
    """Non-streaming WAV: one POST -> full body. Returns (pcm_list, per-item)."""
    pcm_list, per_item = [], []
    sess = session()
    for utt, wav, text in items:
        payload = {
            "model": model,
            "input": text,
            "ref_audio": ref_data_url(wav),
            "response_format": "wav",
            "max_new_tokens": max_new_tokens,
        }
        t0 = time.monotonic()
        r = sess.post(url, json=payload, timeout=600)
        wall = time.monotonic() - t0
        assert r.status_code == 200, (utt, r.status_code, r.text[:200])
        audio, sr = sf.read(io.BytesIO(r.content), dtype="float32")
        assert sr == 24000 and audio.ndim == 1 and audio.size % 3200 == 0
        pcm = to_pcm16(audio)
        pcm_list.append(pcm)
        per_item.append(
            {
                "utterance_id": utt,
                "audio_seconds": audio.size / sr,
                "wall_s": wall,
                "rtf": wall / max(audio.size / sr, 1e-6),
            }
        )
    return pcm_list, per_item


def run_streaming(url, model, items, max_new_tokens):
    """Streaming PCM (stream_format=audio): TTFP = time to first chunk."""
    pcm_list, per_item = [], []
    sess = session()
    for utt, wav, text in items:
        payload = {
            "model": model,
            "input": text,
            "ref_audio": ref_data_url(wav),
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",
            "max_new_tokens": max_new_tokens,
        }
        t0 = time.monotonic()
        r = sess.post(url, json=payload, timeout=600, stream=True)
        assert r.status_code == 200, (utt, r.status_code, r.text[:200])
        chunks = []
        ttfp = None
        for chunk in r.iter_content(chunk_size=8192):
            if ttfp is None and chunk:
                ttfp = time.monotonic() - t0
            if chunk:
                chunks.append(chunk)
        wall = time.monotonic() - t0
        raw = b"".join(chunks)
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        pcm_list.append(raw)
        per_item.append(
            {
                "utterance_id": utt,
                "audio_seconds": audio.size / 24000,
                "wall_s": wall,
                "rtf": wall / max(audio.size / 24000, 1e-6),
                "ttfp_s": ttfp if ttfp is not None else None,
            }
        )
    return pcm_list, per_item


def summarize(per_item, *, has_ttfp=False):
    rtf = [d["rtf"] for d in per_item]
    wall = [d["wall_s"] for d in per_item]
    secs = [d["audio_seconds"] for d in per_item]
    out = {
        "n": len(per_item),
        "mean_rtf": statistics.fmean(rtf) if rtf else None,
        "mean_wall_s": statistics.fmean(wall) if wall else None,
        "total_audio_s": sum(secs),
        "aggregate_rtf": sum(wall) / max(sum(secs), 1e-6) if secs else None,
    }
    if has_ttfp:
        ttfp = [d["ttfp_s"] for d in per_item if d["ttfp_s"] is not None]
        out.update(
            {
                "mean_ttfp_ms": (statistics.fmean(ttfp) * 1000) if ttfp else None,
                "p50_ttfp_ms": (percentile(ttfp, 50) * 1000) if ttfp else None,
                "p99_ttfp_ms": (percentile(ttfp, 99) * 1000) if ttfp else None,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vibevoice/VibeVoice-1.5B-hf")
    parser.add_argument("--dataset", required=True, type=Path, help="locale dir, e.g. .../Seed-TTS-Eval/en")
    parser.add_argument("--items", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--modes", nargs="+", default=["nonstreaming", "streaming"])
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output dir must not exist: {args.output}")
    args.output.mkdir(parents=True)

    items = load_items(args.dataset, args.items, args.seed)
    locale = args.dataset.name.lower()

    deploy_yaml = get_deploy_config_path("vibevoice.yaml")
    with OmniServer(
        args.model,
        ["--deploy-config", deploy_yaml, "--disable-log-stats", "--stage-init-timeout", "600"],
    ) as server:
        url = f"http://{server.host}:{server.port}/v1/audio/speech"
        # Warmup with a short no-reference + short reference request per mode.
        warm_items = items[: args.warmup]
        print(f"warmup {len(warm_items)} items ...", flush=True)
        if "nonstreaming" in args.modes:
            run_nonstreaming(url, args.model, warm_items, 256)
        if "streaming" in args.modes:
            run_streaming(url, args.model, warm_items, 256)

        report = {
            "model": args.model,
            "dataset": str(args.dataset),
            "locale": locale,
            "items": len(items),
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "warmup": args.warmup,
            "modes": {},
            "note": (
                "HF generate() does not stream audio (transformers "
                "VibeVoiceForConditionalGeneration._sample returns audio at end); "
                "client TTFP is only measurable for the vllm-omni online path. "
                "Compare WER/RTF against PR1 gen_hf_seedtts.py outputs."
            ),
        }
        # Collect PCM + perf during the server run; eval after teardown so
        # Whisper/WavLM never coexist with the served model (mirrors PR1).
        collected = {}
        for mode in args.modes:
            print(f"\n=== run {mode} ({len(items)} items) ===", flush=True)
            if mode == "nonstreaming":
                pcm_list, per_item = run_nonstreaming(url, args.model, items, args.max_tokens)
            elif mode == "streaming":
                pcm_list, per_item = run_streaming(url, args.model, items, args.max_tokens)
            else:
                raise SystemExit(f"unknown mode {mode}")
            collected[mode] = (pcm_list, per_item)
            # Save generated audio for offline re-eval / listening.
            wav_dir = args.output / mode
            wav_dir.mkdir(exist_ok=True)
            for (utt, _, _), pcm in zip(items, pcm_list):
                a = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                sf.write(wav_dir / f"{utt}.wav", a, 24000, subtype="FLOAT")

    # Eval after the engine is released so Whisper/WavLM don't coexist with it.
    for mode in args.modes:
        pcm_list, per_item = collected[mode]
        print(f"\n=== eval {mode} ({len(items)} items) ===", flush=True)
        metrics = eval_pcm(items, pcm_list, locale)
        print_seed_tts_wer_summary(metrics)
        summ = summarize(per_item, has_ttfp=(mode == "streaming"))
        print(json.dumps(summ, indent=2), flush=True)
        report["modes"][mode] = {"perf": summ, "wer_sim": metrics, "per_item": per_item}

    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n-> {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
