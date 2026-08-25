# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage for the opt-in VibeVoice quality driver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e.accuracy.vibevoice.run_vibevoice_quality import (
    _run_locale,
    _validate_result,
    build_arg_parser,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _scored_result() -> dict:
    return {
        "seed_tts_content_evaluated": 2,
        "seed_tts_content_error_mean": 0.1,
        "seed_tts_request_failed": 0,
        "seed_tts_no_pcm": 0,
        "seed_tts_asr_failed": 0,
        "seed_tts_sim_evaluated": 2,
        "seed_tts_sim_mean": 0.8,
        "seed_tts_sim_failed": 0,
        "seed_tts_sim_skipped_no_ref": 0,
        "seed_tts_utmos_evaluated": 2,
        "seed_tts_utmos_mean": 3.5,
        "seed_tts_utmos_failed": 0,
        "seed_tts_finish_reason_counts": {"stop": 2},
        "seed_tts_length_capped": 0,
        "seed_tts_terminal_stop_required": 2,
        "seed_tts_required_stop_count": 2,
        "seed_tts_missing_finish_reason": 0,
        "seed_tts_non_stop_excluded": 0,
        "seed_tts_total_requests": 2,
    }


def test_validate_result_applies_locale_metrics_and_thresholds() -> None:
    assert not _validate_result(
        _scored_result(),
        locale="en",
        min_evaluated=2,
        max_mean_content_error=0.2,
        min_mean_sim=0.7,
        sim_enabled=True,
    )

    errors = _validate_result(
        _scored_result(),
        locale="zh",
        min_evaluated=3,
        max_mean_content_error=0.05,
        min_mean_sim=0.9,
        sim_enabled=True,
    )
    assert "content evaluated=2 < required 3" in errors
    assert "mean CER=0.100000 > 0.050000" in errors
    assert "mean speaker similarity=0.8 < 0.900000" in errors

    length_result = _scored_result()
    length_result["seed_tts_finish_reason_counts"] = {"stop": 1, "length": 1}
    length_result["seed_tts_length_capped"] = 1
    length_result["seed_tts_required_stop_count"] = 1
    length_result["seed_tts_non_stop_excluded"] = 1
    errors = _validate_result(
        length_result,
        locale="en",
        min_evaluated=2,
        max_mean_content_error=None,
        min_mean_sim=None,
        sim_enabled=False,
    )
    assert "terminal stops=1 != required 2" in errors
    assert "non-stop samples excluded=1" in errors
    assert "length-capped quality requests=1" in errors
    assert "unexpected finish reasons=['length']" in errors

    incomplete_result = _scored_result()
    incomplete_result["seed_tts_required_stop_count"] = 1
    incomplete_result["seed_tts_missing_finish_reason"] = 1
    incomplete_result["seed_tts_sim_evaluated"] = 1
    incomplete_result["seed_tts_sim_failed"] = 1
    errors = _validate_result(
        incomplete_result,
        locale="en",
        min_evaluated=2,
        max_mean_content_error=None,
        min_mean_sim=None,
        sim_enabled=True,
    )
    assert "missing finish reasons=1" in errors
    assert "speaker similarity evaluated=1 != content evaluated 2" in errors
    assert "speaker similarity failures=1" in errors

    utmos_result = _scored_result()
    utmos_result["seed_tts_utmos_evaluated"] = 1
    utmos_result["seed_tts_utmos_failed"] = 1
    utmos_result["seed_tts_utmos_mean"] = 2.9
    errors = _validate_result(
        utmos_result,
        locale="en",
        min_evaluated=2,
        max_mean_content_error=None,
        min_mean_sim=None,
        sim_enabled=False,
        min_mean_utmos=3.0,
        utmos_enabled=True,
    )
    assert "UTMOS evaluated=1 != content evaluated 2" in errors
    assert "UTMOS failures=1" in errors
    assert "mean UTMOS=2.9 < 3.000000" in errors


def test_utmos_loader_honors_pinned_revision(monkeypatch) -> None:
    import huggingface_hub
    import torch

    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    calls = {}

    class FakeModel:
        def eval(self):
            return self

    def fake_download(**kwargs):
        calls.update(kwargs)
        return "/tmp/utmos.jit"

    fake_model = FakeModel()
    monkeypatch.setenv("SEED_TTS_UTMOS_HF_REVISION", "fixed-revision")
    monkeypatch.setattr(seed_tts_eval, "_utmos_jit_model", None)
    monkeypatch.setattr(seed_tts_eval, "_utmos_jit_load_failed", False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.jit, "load", lambda path, map_location: fake_model)

    assert seed_tts_eval._ensure_utmos_jit_model() is fake_model
    assert calls == {
        "repo_id": "balacoon/utmos",
        "filename": "utmos.jit",
        "repo_type": "model",
        "revision": "fixed-revision",
    }


def test_min_evaluated_defaults_to_requested_prompt_count() -> None:
    args = build_arg_parser().parse_args(["--num-prompts", "7"])

    assert args.min_evaluated is None


def test_run_locale_selects_vibevoice_schema_and_sse_transport(tmp_path: Path, monkeypatch) -> None:
    args = build_arg_parser().parse_args(
        [
            "--locale",
            "en",
            "--dataset-path",
            str(tmp_path / "dataset"),
            "--result-dir",
            str(tmp_path / "results"),
            "--save-audio-dir",
            str(tmp_path / "audio"),
            "--num-prompts",
            "2",
            "--enable-utmos",
            "--min-mean-utmos",
            "3.0",
        ]
    )
    captured = {}

    def fake_run(command, *, env, check):
        assert check is True
        captured["command"] = command
        captured["env"] = env
        filename = command[command.index("--result-filename") + 1]
        result_dir = Path(command[command.index("--result-dir") + 1])
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / filename).write_text(json.dumps(_scored_result()), encoding="utf-8")

    monkeypatch.setattr("tests.e2e.accuracy.vibevoice.run_vibevoice_quality.subprocess.run", fake_run)

    result_path, errors = _run_locale(args, "en", "vllm")

    assert result_path.is_file()
    assert not errors
    command = captured["command"]
    assert command[command.index("--backend") + 1] == "openai-audio-speech"
    assert command[command.index("--dataset-name") + 1] == "seed-tts-vibevoice"
    assert command[command.index("--max-concurrency") + 1] == "4"
    extra_body = json.loads(command[command.index("--extra-body") + 1])
    assert extra_body == {
        "stream": True,
        "stream_format": "sse",
        "response_format": "pcm",
    }
    assert captured["env"]["SEED_TTS_SIM_EVAL"] == "1"
    assert captured["env"]["SEED_TTS_UTMOS_EVAL"] == "1"
    assert captured["env"]["SEED_TTS_UTMOS_HF_REVISION"] == "b44d848644aa5d89a6e4f180ea50452d8c162db2"
    assert captured["env"]["SEED_TTS_WER_SAVE_AUDIO_DIR"].endswith("/en")
