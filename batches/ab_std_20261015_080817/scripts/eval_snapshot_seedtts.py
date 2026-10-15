# Offline VibeVoice Seed-TTS-Eval quality/perf probe (local evidence only, not part of the PR).
# Run under .venv-test (Omni env, transformers 5.14.1). Reuses the framework harness:
#   SeedTTSDataset (selection) + compute_seed_tts_wer_metrics (WER/SIM via whisper-large-v3 / wavlm-base-plus).
# Two modes:
#   default      : Omni generate + eval (engine closed before eval so Whisper/WavLM don't coexist).
#   --eval-wavs D: eval WAVs in D (e.g. HF outputs from gen_hf_seedtts.py) with the same harness.
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from vllm import SamplingParams

from vllm_omni.benchmarks.data_modules.seed_tts_dataset import SeedTTSDataset, SeedTTSSampleRequest
from vllm_omni.benchmarks.data_modules.seed_tts_eval import compute_seed_tts_wer_metrics, print_seed_tts_wer_summary
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_executor.models.vibevoice.prompt import render_vibevoice_prompt


@dataclass
class _Out:
    success: bool = True
    error: str = ""
    tts_output_pcm_bytes: bytes = b""


def load_items(locale_dir: Path, count: int, seed: int):
    ds = SeedTTSDataset(dataset_path=str(locale_dir.parent), random_seed=seed, locale=locale_dir.name)
    rows = ds.data
    if count > len(rows):
        raise SystemExit(f"requested {count} items but meta.lst has {len(rows)}")
    return [(r.utterance_id, locale_dir / r.prompt_wav_rel, r.target_text) for r in rows[:count]]


def to_pcm16(audio) -> bytes:
    return (
        (torch.as_tensor(audio).detach().cpu().float().reshape(-1).clamp(-1.0, 1.0).numpy() * 32767.0)
        .astype("<i2")
        .tobytes()
    )


def eval_pcm(items, pcm_list, locale):
    requests = [
        SeedTTSSampleRequest(
            prompt=text,
            prompt_len=0,
            seed_tts_utterance_id=utt,
            seed_tts_locale=locale,
            seed_tts_ref_wav_path=str(wav),
        )
        for utt, wav, text in items
    ]
    return compute_seed_tts_wer_metrics(
        requests, [_Out(tts_output_pcm_bytes=p) for p in pcm_list], include_per_item=True
    )


def eval_wav_dir(args):
    items = load_items(args.dataset, args.items, args.seed)
    locale = args.dataset.name.lower()
    pcm = []
    for utt, _, _ in items:
        name = f"r{args.round}_{utt}.wav" if args.round is not None else f"{utt}.wav"
        audio, sr = sf.read(args.eval_wavs / name, dtype="float32")
        assert sr == 24000, f"{name}: expected 24kHz, got {sr}"
        pcm.append((np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
    metrics = eval_pcm(items, pcm, locale)
    print_seed_tts_wer_summary(metrics)
    report = {"backend": "wav-dir", "wav_dir": str(args.eval_wavs), "items": len(items), **metrics}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"-> {args.output / 'report.json'}")


def _input_order(outputs):
    """Omni yields by completion order; sort by the request_id input index prefix."""
    return sorted(outputs, key=lambda o: int(o.request_id.split("_")[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vibevoice/VibeVoice-1.5B-hf")
    parser.add_argument("--dataset", required=True, type=Path, help="locale dir, e.g. .../Seed-TTS-Eval/en")
    parser.add_argument("--items", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--save-wavs", type=Path, default=None)
    parser.add_argument("--eval-wavs", type=Path, default=None, help="eval WAVs in this dir instead of generating")
    parser.add_argument("--round", type=int, default=None, help="round prefix for --eval-wavs (r{round}_{utt}.wav)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.eval_wavs:
        raise SystemExit(f"output dir must not exist: {args.output}")
    if args.save_wavs is not None:
        args.save_wavs.mkdir(parents=True, exist_ok=True)

    items = load_items(args.dataset, args.items, args.seed)
    if args.dry_run:
        print(f"selected {len(items)} items (seed={args.seed}); first: {[i[0] for i in items[:3]]}")
        return
    if args.eval_wavs is not None:
        eval_wav_dir(args)
        return

    locale = args.dataset.name.lower()
    # Single SamplingParams per stage (VibeVoice is single-stage); one stage params applies to the whole batch.
    sp = [SamplingParams(max_tokens=args.max_tokens)]
    omni = Omni(model=args.model, seed=args.seed)
    # (round, pcm_list, wall, audio_s, stops)
    gen_results = []
    total_wall = total_audio = total_stops = 0.0
    try:
        for _ in range(args.warmup):
            warm = items[: args.batch_size]
            omni.generate(
                [
                    {"prompt": render_vibevoice_prompt(t), "multi_modal_data": {"audio": sf.read(w, dtype="float32")}}
                    for _, w, t in warm
                ],
                sp,
            )
        for rnd in range(args.rounds):
            rnd_pcm, rnd_wall, rnd_audio, rnd_stops = [], 0.0, 0.0, 0
            for start in range(0, len(items), args.batch_size):
                batch = items[start : start + args.batch_size]
                prompts = [
                    {
                        "prompt": render_vibevoice_prompt(text),
                        "multi_modal_data": {"audio": sf.read(wav, dtype="float32")},
                    }
                    for _, wav, text in batch
                ]
                t0 = time.perf_counter()
                outputs = _input_order(omni.generate(prompts, sp))
                rnd_wall += time.perf_counter() - t0
                for (utt, _, _), output in zip(batch, outputs, strict=True):
                    c = output.outputs[0]
                    audio = c.multimodal_output["audio"]
                    if isinstance(audio, list):
                        audio = torch.cat([torch.as_tensor(x).reshape(-1) for x in audio])
                    rnd_stops += c.finish_reason == "stop"
                    rnd_audio += audio.numel() / 24000
                    chunk = to_pcm16(audio)
                    rnd_pcm.append(chunk)
                    if args.save_wavs is not None:
                        sf.write(
                            args.save_wavs / (f"r{rnd}_{utt}.wav" if args.rounds > 1 else f"{utt}.wav"),
                            audio.float().cpu().numpy(),
                            24000,
                            subtype="FLOAT",
                        )
            total_wall += rnd_wall
            total_audio += rnd_audio
            total_stops += rnd_stops
            gen_results.append((rnd, rnd_pcm, rnd_wall, rnd_audio, rnd_stops))
    finally:
        omni.close()

    # Eval after the engine is released so Whisper/WavLM don't coexist with the Omni engine.
    rounds_metrics = []
    for rnd, rnd_pcm, rnd_wall, rnd_audio, rnd_stops in gen_results:
        metrics = eval_pcm(items, rnd_pcm, locale)
        rounds_metrics.append(metrics)
        print_seed_tts_wer_summary(metrics)
        print(
            f"round {rnd}: wall={rnd_wall:.1f}s audio={rnd_audio:.1f}s "
            f"rtf={rnd_wall / rnd_audio:.4f} stop={rnd_stops}/{len(items)}"
        )

    report = {
        "backend": "omni",
        "model": args.model,
        "dataset": str(args.dataset),
        "items": len(items),
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "natural_stop_rate": total_stops / (len(items) * args.rounds),
        "generate_wall_s": total_wall,
        "audio_s": total_audio,
        "aggregate_rtf": total_wall / total_audio,
        "rounds_metrics": rounds_metrics,
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(
        f"aggregate_rtf={report['aggregate_rtf']:.4f} "
        f"natural_stop={int(total_stops)}/{len(items) * args.rounds} -> {args.output}"
    )


if __name__ == "__main__":
    main()
