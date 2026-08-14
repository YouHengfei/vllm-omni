# SPDX-License-Identifier: Apache-2.0
"""Tests for normalized binary Speech responses in the E2E client helper."""

from __future__ import annotations

import time
from types import SimpleNamespace

from tests.helpers.runtime import OpenAIClientHandler


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
