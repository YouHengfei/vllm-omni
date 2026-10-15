# HF Transformers 5.19 reference generation -> WAVs (local evidence only, not part of the PR).
# Run under the HF env (/SharedData/youhf/tmp/vibevoice-pr1-hf-batch2). Eval afterwards via
# eval_snapshot_seedtts.py --eval-wavs under .venv-test so ASR/WavLM match the Omni path exactly.
# Config matches the adapter: CFG 1.3, 10 diffusion steps, BF16, greedy, max_tokens 2048, same 100-item subset.
import argparse
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForTextToWaveform, AutoProcessor


def load_items(locale_dir: Path, count: int, seed: int):
    # HF env has no vllm_omni, so this mirrors SeedTTSDataset.load_data (parse + shuffle);
    # selection is verified identical to the framework loader.
    rows = []
    for line in (locale_dir / "meta.lst").read_text().splitlines():
        parts = line.split("|")
        if len(parts) >= 4:
            rows.append((parts[0], locale_dir / parts[2], parts[3]))
    random.Random(seed).shuffle(rows)
    if count > len(rows):
        raise SystemExit(f"requested {count} items but meta.lst has {len(rows)}")
    return rows[:count]


def run(model, proc, batch, max_tokens):
    convs = [
        [
            {
                "role": "0",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "audio", "path": sf.read(wav, dtype="float32")[0]},
                ],
            }
        ]
        for _, wav, text in batch
    ]
    # Time the whole per-batch path (prompt proc + generate + D2H) to match the Omni generate() boundary,
    # which includes ingress and returns CPU audio. Sync after generate so wall reflects GPU completion.
    t0 = time.perf_counter()
    inputs = proc.apply_chat_template(
        convs, tokenize=True, return_dict=True, padding=True, add_generation_prompt=True
    ).to(model.device, model.dtype)
    with torch.inference_mode():
        res = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            guidance_scale=1.3,
            num_diffusion_steps=10,
            return_dict_in_generate=True,
        )
    torch.accelerator.synchronize()
    eos = model.config.eos_token_id
    audios, stops, audio_s = [], 0, 0.0
    gen = res.sequences[:, inputs["input_ids"].shape[1] :]
    for audio, g in zip(res.audio, gen, strict=True):
        audio = np.zeros(0, np.float32) if audio is None else audio.float().cpu().numpy().reshape(-1)
        stops += bool((g == eos).any())
        audio_s += audio.size / 24000
        audios.append(audio)
    wall = time.perf_counter() - t0
    return wall, audios, stops, audio_s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, type=Path, help="locale dir, e.g. .../Seed-TTS-Eval/en")
    parser.add_argument("--items", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path, help="wav output dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    items = load_items(args.dataset, args.items, args.seed)
    if args.dry_run:
        print(f"selected {len(items)} items (seed={args.seed}); first: {[i[0] for i in items[:3]]}")
        return

    proc = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForTextToWaveform.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    ).eval()
    torch.manual_seed(args.seed)
    for _ in range(args.warmup):
        run(model, proc, items[: args.batch_size], args.max_tokens)

    gen_wall = audio_s = 0.0
    stops = 0
    for start in range(0, len(items), args.batch_size):
        batch = items[start : start + args.batch_size]
        wall, audios, sts, a_s = run(model, proc, batch, args.max_tokens)
        gen_wall += wall
        audio_s += a_s
        stops += sts
        for (utt, _, _), audio in zip(batch, audios, strict=True):
            sf.write(args.output / f"{utt}.wav", audio, 24000, subtype="FLOAT")
    print(
        f"HF gen done: wall={gen_wall:.1f}s audio={audio_s:.1f}s "
        f"rtf={gen_wall / max(audio_s, 1e-6):.4f} stop={stops}/{len(items)} -> {args.output}"
    )


if __name__ == "__main__":
    main()
