#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Opt-in VibeVoice Seed-TTS WER/CER and speaker-similarity evaluation.

This client expects an already-running VibeVoice OpenAI server. It reuses the
repository's pinned Seed-TTS judges while selecting VibeVoice's supported
reference-audio-only request schema and SSE transport. Thresholds default to
``None`` so the first runs establish a baseline instead of inventing a gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.e2e.accuracy.qwen3_omni.qwen3_omni_acc_bench_core import (
    DEFAULT_SEED_TTS_HF_REPO,
    find_vllm_cli,
    load_benchmark_result,
)


def _default_result_dir() -> Path:
    return Path(__file__).resolve().parent / "results"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("VIBEVOICE_QUALITY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VIBEVOICE_QUALITY_PORT", "8000")))
    parser.add_argument(
        "--model",
        default=os.environ.get("VIBEVOICE_QUALITY_MODEL", "VibeVoice"),
        help="Model name exposed by the running server.",
    )
    parser.add_argument(
        "--dataset-path",
        default=os.environ.get("VIBEVOICE_QUALITY_DATASET", DEFAULT_SEED_TTS_HF_REPO),
        help="Seed-TTS local root or Hugging Face dataset id.",
    )
    parser.add_argument(
        "--seed-tts-root",
        type=Path,
        default=(Path(os.environ["SEED_TTS_ROOT"]) if os.environ.get("SEED_TTS_ROOT") else None),
    )
    parser.add_argument("--locale", choices=("en", "zh", "both"), default="both")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--output-len", type=int, default=2048)
    parser.add_argument("--result-dir", type=Path, default=_default_result_dir())
    parser.add_argument("--save-audio-dir", type=Path)
    parser.add_argument("--eval-device", default=os.environ.get("SEED_TTS_EVAL_DEVICE", "cpu"))
    parser.add_argument("--sim-device", default=os.environ.get("SEED_TTS_SIM_DEVICE", "cpu"))
    parser.add_argument("--disable-sim", action="store_true")
    parser.add_argument("--enable-utmos", action="store_true")
    parser.add_argument("--min-evaluated", type=int, default=1)
    parser.add_argument("--max-mean-content-error", type=float)
    parser.add_argument("--min-mean-sim", type=float)
    parser.add_argument("--ready-check-timeout-sec", type=int, default=300)
    return parser


def _validate_result(
    result: dict[str, Any],
    *,
    locale: str,
    min_evaluated: int,
    max_mean_content_error: float | None,
    min_mean_sim: float | None,
    sim_enabled: bool,
    utmos_enabled: bool = False,
) -> list[str]:
    errors: list[str] = []
    if setup_error := result.get("seed_tts_eval_setup_error"):
        return [f"judge setup failed: {setup_error}"]

    evaluated = int(result.get("seed_tts_content_evaluated", 0) or 0)
    if evaluated < min_evaluated:
        errors.append(f"content evaluated={evaluated} < required {min_evaluated}")
    for key in ("seed_tts_request_failed", "seed_tts_no_pcm", "seed_tts_asr_failed"):
        count = int(result.get(key, 0) or 0)
        if count:
            errors.append(f"{key}={count}")

    finish_reason_counts = result.get("seed_tts_finish_reason_counts") or {}
    terminal_stop_required = int(
        result.get(
            "seed_tts_terminal_stop_required",
            result.get("seed_tts_total_requests", evaluated),
        )
        or 0
    )
    required_stop_count = int(
        result.get(
            "seed_tts_required_stop_count",
            finish_reason_counts.get("stop", 0),
        )
        or 0
    )
    if required_stop_count != terminal_stop_required:
        errors.append(f"terminal stops={required_stop_count} != required {terminal_stop_required}")
    missing_finish_reason = int(result.get("seed_tts_missing_finish_reason", 0) or 0)
    if missing_finish_reason:
        errors.append(f"missing finish reasons={missing_finish_reason}")
    non_stop_excluded = int(result.get("seed_tts_non_stop_excluded", 0) or 0)
    if non_stop_excluded:
        errors.append(f"non-stop samples excluded={non_stop_excluded}")
    length_capped = int(result.get("seed_tts_length_capped", 0) or 0)
    if length_capped:
        errors.append(f"length-capped quality requests={length_capped}")
    unexpected_finish_reasons = sorted(reason for reason in finish_reason_counts if reason != "stop")
    if unexpected_finish_reasons:
        errors.append(f"unexpected finish reasons={unexpected_finish_reasons}")

    mean_error = result.get("seed_tts_content_error_mean")
    if mean_error is None and evaluated:
        errors.append("mean content error is missing")
    elif max_mean_content_error is not None and float(mean_error) > max_mean_content_error:
        metric = "CER" if locale == "zh" else "WER"
        errors.append(f"mean {metric}={float(mean_error):.6f} > {max_mean_content_error:.6f}")

    if sim_enabled:
        sim_evaluated = int(result.get("seed_tts_sim_evaluated", 0) or 0)
        if sim_evaluated != evaluated:
            errors.append(f"speaker similarity evaluated={sim_evaluated} != content evaluated {evaluated}")
        sim_failed = int(result.get("seed_tts_sim_failed", 0) or 0)
        if sim_failed:
            errors.append(f"speaker similarity failures={sim_failed}")
        sim_skipped = int(result.get("seed_tts_sim_skipped_no_ref", 0) or 0)
        if sim_skipped:
            errors.append(f"speaker similarity missing references={sim_skipped}")
        mean_sim = result.get("seed_tts_sim_mean")
        if min_mean_sim is not None and (mean_sim is None or float(mean_sim) < min_mean_sim):
            errors.append(f"mean speaker similarity={mean_sim!r} < {min_mean_sim:.6f}")
    if utmos_enabled:
        utmos_evaluated = int(result.get("seed_tts_utmos_evaluated", 0) or 0)
        if utmos_evaluated != evaluated:
            errors.append(f"UTMOS evaluated={utmos_evaluated} != content evaluated {evaluated}")
        utmos_failed = int(result.get("seed_tts_utmos_failed", 0) or 0)
        if utmos_failed:
            errors.append(f"UTMOS failures={utmos_failed}")
    return errors


def _run_locale(args: argparse.Namespace, locale: str, vllm_cli: str) -> tuple[Path, list[str]]:
    args.result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"vibevoice_quality_{locale}_{timestamp}.json"
    command = [
        vllm_cli,
        "bench",
        "serve",
        "--omni",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--endpoint",
        "/v1/audio/speech",
        "--backend",
        "openai-audio-speech",
        "--dataset-name",
        "seed-tts-vibevoice",
        "--dataset-path",
        args.dataset_path,
        "--seed-tts-locale",
        locale,
        "--output-len",
        str(args.output_len),
        "--num-prompts",
        str(args.num_prompts),
        "--max-concurrency",
        str(args.max_concurrency),
        "--request-rate",
        "inf",
        "--no-oversample",
        "--seed-tts-wer-eval",
        "--seed-tts-wer-save-items",
        "--extra-body",
        json.dumps(
            {
                "stream": True,
                "stream_format": "sse",
                "response_format": "pcm",
            },
            separators=(",", ":"),
        ),
        "--percentile-metrics",
        "e2el,audio_ttfp,audio_rtf",
        "--ready-check-timeout-sec",
        str(args.ready_check_timeout_sec),
        "--save-result",
        "--result-dir",
        str(args.result_dir),
        "--result-filename",
        filename,
    ]
    if args.seed_tts_root is not None:
        command.extend(["--seed-tts-root", str(args.seed_tts_root.expanduser().resolve())])

    env = os.environ.copy()
    env.update(
        {
            "SEED_TTS_WER_EVAL": "1",
            "SEED_TTS_SIM_EVAL": "0" if args.disable_sim else "1",
            "SEED_TTS_UTMOS_EVAL": "1" if args.enable_utmos else "0",
            "SEED_TTS_EVAL_DEVICE": args.eval_device,
            "SEED_TTS_SIM_DEVICE": args.sim_device,
            "VLLM_OMNI_BENCH_SPEECH_STREAM_FORMAT": "sse",
            "VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE": "24000",
            "VLLM_OMNI_BENCH_AUDIO_CHANNELS": "1",
        }
    )
    if args.save_audio_dir is not None:
        locale_dir = args.save_audio_dir.expanduser().resolve() / locale
        env["SEED_TTS_WER_SAVE_AUDIO_DIR"] = str(locale_dir)

    print("\n$", " ".join(command), "\n", flush=True)
    subprocess.run(command, env=env, check=True)
    result_path = args.result_dir / filename
    result = load_benchmark_result(result_path)
    errors = _validate_result(
        result,
        locale=locale,
        min_evaluated=args.min_evaluated,
        max_mean_content_error=args.max_mean_content_error,
        min_mean_sim=args.min_mean_sim,
        sim_enabled=not args.disable_sim,
        utmos_enabled=args.enable_utmos,
    )
    metric_name = "CER" if locale == "zh" else "WER"
    print(
        f"[{locale}] {metric_name} mean={result.get('seed_tts_content_error_mean')} "
        f"SIM mean={result.get('seed_tts_sim_mean')} result={result_path}",
        flush=True,
    )
    return result_path, errors


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.num_prompts < 1:
        raise ValueError("--num-prompts must be positive")
    if not 1 <= args.max_concurrency <= 2:
        raise ValueError("--max-concurrency must be 1 or 2 for the fixed VibeVoice deployment")
    if args.output_len < 1:
        raise ValueError("--output-len must be positive")

    locales = ("en", "zh") if args.locale == "both" else (args.locale,)
    vllm_cli = find_vllm_cli()
    failures: list[str] = []
    for locale in locales:
        _, errors = _run_locale(args, locale, vllm_cli)
        failures.extend(f"[{locale}] {error}" for error in errors)
    if failures:
        print("\nVibeVoice quality evaluation failed:", flush=True)
        for failure in failures:
            print(f"- {failure}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
