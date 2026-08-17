# SPDX-License-Identifier: Apache-2.0
"""Tests for normalized binary Speech responses in the E2E client helper."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from tests.helpers import runtime as runtime_helpers
from tests.helpers.runtime import (
    OmniRunner,
    OmniServerParams,
    OpenAIClientHandler,
    iter_omni_server,
)
from tests.helpers.stage_config import (
    get_deploy_config_path,
    stage_config_path_for_run_level,
    stage_config_uses_dummy_load_format,
)


class _BinarySpeechResponse:
    def __init__(self, payload: bytes, headers: dict[str, str]) -> None:
        self._payload = payload
        self.response = SimpleNamespace(headers=headers)

    def read(self) -> bytes:
        return self._payload


class _StreamingSpeechResponse(_BinarySpeechResponse):
    def iter_bytes(self):
        yield self._payload[:2]
        yield self._payload[2:]


def test_real_weight_server_guard_skips_before_core_model_startup() -> None:
    request = SimpleNamespace(
        param=OmniServerParams(
            model="test-only-model",
            require_real_weights=True,
        )
    )
    server = iter_omni_server(
        request,
        run_level="core_model",
        model_prefix="",
        omni_fixture_lock=threading.Lock(),
    )

    with pytest.raises(pytest.skip.Exception, match="requires real weights"):
        next(server)


def test_vibevoice_real_weight_stage_config_guard_distinguishes_run_levels() -> None:
    deploy = get_deploy_config_path("vibevoice.yaml")
    core_config = stage_config_path_for_run_level(deploy, "core_model")
    advanced_config = stage_config_path_for_run_level(deploy, "advanced_model")

    assert core_config is not None
    assert advanced_config is not None
    assert stage_config_uses_dummy_load_format(core_config) is True
    assert stage_config_uses_dummy_load_format(advanced_config) is False


def test_omni_runner_residual_cleanup_only_snapshots_own_engine_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SimpleNamespace(
        pid=101,
        cmdline=lambda: ["vllm::EngineCore"],
        name=lambda: "python",
    )
    unrelated = SimpleNamespace(
        pid=202,
        cmdline=lambda: ["python", "unrelated.py"],
        name=lambda: "python",
    )
    root = SimpleNamespace(children=lambda recursive: [engine, unrelated])
    monkeypatch.setattr(runtime_helpers.psutil, "Process", lambda _pid: root)

    runner = object.__new__(OmniRunner)

    assert runner._owned_engine_processes() == [engine]


def test_local_openai_client_does_not_inherit_proxy_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:7890")

    handler = OpenAIClientHandler(port=12345)
    try:
        assert handler.client._client._trust_env is False
    finally:
        handler.client.close()


def test_non_stream_speech_response_preserves_lower_cased_headers() -> None:
    response = _BinarySpeechResponse(
        b"RIFF",
        {
            "Content-Type": "audio/wav",
            "X-Finish-Reason": "stop",
        },
    )

    result = OpenAIClientHandler._process_non_stream_audio_speech_response(
        None,
        response,
        wall_start=time.perf_counter(),
    )

    assert result.success is True
    assert result.audio_bytes == b"RIFF"
    assert result.audio_format == "audio/wav"
    assert result.response_headers == {
        "content-type": "audio/wav",
        "x-finish-reason": "stop",
    }


def test_stream_speech_response_preserves_lower_cased_headers() -> None:
    response = _StreamingSpeechResponse(
        b"\x01\x02\x03\x04",
        {"Content-Type": "audio/pcm"},
    )

    result = OpenAIClientHandler._process_stream_audio_speech_response(
        None,
        response,
        response_format="pcm",
        wall_start=time.perf_counter(),
    )

    assert result.success is True
    assert result.audio_bytes == b"\x01\x02\x03\x04"
    assert result.audio_format == "audio/pcm"
    assert result.response_headers == {"content-type": "audio/pcm"}
