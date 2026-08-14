# SPDX-License-Identifier: Apache-2.0
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
        "seed_tts_finish_reason_counts": {"stop": 2},
        "seed_tts_length_capped": 0,
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
    errors = _validate_result(
        length_result,
        locale="en",
        min_evaluated=2,
        max_mean_content_error=None,
        min_mean_sim=None,
        sim_enabled=False,
    )
    assert "length-capped quality requests=1" in errors
    assert "unexpected finish reasons=['length']" in errors


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
    extra_body = json.loads(command[command.index("--extra-body") + 1])
    assert extra_body == {
        "stream": True,
        "stream_format": "sse",
        "response_format": "pcm",
    }
    assert captured["env"]["VLLM_OMNI_BENCH_SPEECH_STREAM_FORMAT"] == "sse"
    assert captured["env"]["SEED_TTS_SIM_EVAL"] == "1"
    assert captured["env"]["SEED_TTS_WER_SAVE_AUDIO_DIR"].endswith("/en")
