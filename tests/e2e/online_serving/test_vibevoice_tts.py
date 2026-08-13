# SPDX-License-Identifier: Apache-2.0
"""Real OpenAI speech HTTP coverage for the official VibeVoice checkpoint."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path, load_test_audio_data_url
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.tts,
    pytest.mark.skipif(
        not torch.accelerator.is_available() or torch.accelerator.device_count() < 2,
        reason="Two CUDA devices are required",
    ),
]

_MODEL_ROOT = os.getenv("VIBEVOICE_TEST_MODEL_ROOT")
_MODEL = str(Path(_MODEL_ROOT) / "VibeVoice") if _MODEL_ROOT else "VibeVoice"
_TOKENIZER = str(Path(_MODEL_ROOT) / "VibeVoice-1.5B-hf") if _MODEL_ROOT else "VibeVoice-1.5B-hf"
_REFERENCE_DATA_URL = load_test_audio_data_url("cosyvoice3/zero_shot_prompt.wav")
_REFERENCE_FILE = get_asset_path("cosyvoice3/zero_shot_prompt.wav").resolve()


def _data_url_for_wav(waveform: np.ndarray, sample_rate: int = 24_000) -> str:
    with io.BytesIO() as buffer:
        sf.write(buffer, waveform, sample_rate, format="WAV")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


_SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=_MODEL,
            stage_config_path=get_deploy_config_path("vibevoice.yaml"),
            server_args=[
                "--tokenizer",
                _TOKENIZER,
                "--allowed-local-media-path",
                str(get_asset_path("").resolve()),
                "--disable-log-stats",
            ],
            env_dict={"VLLM_USE_FLASHINFER_SAMPLER": "0"},
            init_timeout=900,
            stage_init_timeout=600,
        ),
        id="official-vibevoice",
    )
]


def _speech_url(omni_server) -> str:
    return f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"


def _post(omni_server, payload: dict, *, timeout: float = 300.0) -> httpx.Response:
    # Local E2E traffic must not inherit developer/CI SOCKS proxy settings.
    with httpx.Client(trust_env=False, timeout=timeout) as client:
        return client.post(_speech_url(omni_server), json=payload)


def _assert_error(response: httpx.Response, message: str) -> None:
    assert response.status_code in (400, 422), response.text
    assert message in response.text


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", _SERVER_PARAMS, indirect=True)
def test_vibevoice_http_wav_pcm_and_local_file(omni_server) -> None:
    base_payload = {
        "model": omni_server.model,
        "input": "Hello.",
        "ref_audio": _REFERENCE_DATA_URL,
        "stream": False,
    }

    wav_response = _post(
        omni_server,
        {**base_payload, "response_format": "wav"},
    )
    assert wav_response.status_code == 200, wav_response.text
    assert wav_response.headers["content-type"].startswith("audio/wav")
    waveform, sample_rate = sf.read(io.BytesIO(wav_response.content), dtype="float32")
    assert sample_rate == 24_000
    assert waveform.ndim == 1
    assert waveform.size > 0
    assert waveform.size % 3_200 == 0
    assert np.isfinite(waveform).all()

    pcm_response = _post(
        omni_server,
        {**base_payload, "response_format": "pcm"},
    )
    assert pcm_response.status_code == 200, pcm_response.text
    assert pcm_response.headers["content-type"].startswith("audio/pcm")
    assert len(pcm_response.content) > 0
    assert len(pcm_response.content) % (3_200 * 2) == 0

    truncated_response = _post(
        omni_server,
        {
            **base_payload,
            "response_format": "wav",
            "max_new_tokens": 2,
        },
    )
    assert truncated_response.status_code == 200, truncated_response.text
    assert truncated_response.headers.get("X-Finish-Reason") == "length"
    assert truncated_response.content[:4] == b"RIFF"

    file_response = _post(
        omni_server,
        {
            **base_payload,
            "ref_audio": _REFERENCE_FILE.as_uri(),
            "response_format": "wav",
        },
    )
    assert file_response.status_code == 200, file_response.text
    assert file_response.content[:4] == b"RIFF"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", _SERVER_PARAMS, indirect=True)
def test_vibevoice_http_rejects_invalid_requests_before_generation(omni_server) -> None:
    valid = {
        "model": omni_server.model,
        "input": "Hello.",
        "ref_audio": _REFERENCE_DATA_URL,
        "response_format": "wav",
    }

    _assert_error(
        _post(omni_server, {**valid, "stream_format": "audio"}),
        "non-streaming",
    )
    _assert_error(
        _post(
            omni_server,
            {**valid, "extra_params": {"num_diffusion_steps": 0}},
        ),
        "positive integer",
    )
    _assert_error(
        _post(omni_server, {**valid, "seed": 42}),
        "request-level seed",
    )
    _assert_error(
        _post(omni_server, {**valid, "ref_audio": "not-a-url"}),
        "ref_audio must be a URL",
    )
    _assert_error(
        _post(
            omni_server,
            {
                **valid,
                "ref_audio": "https://127.0.0.1:1/missing.wav",
            },
        ),
        "Cannot connect",
    )

    too_long = _data_url_for_wav(np.zeros(60 * 24_000 + 1, dtype=np.float32))
    _assert_error(
        _post(omni_server, {**valid, "ref_audio": too_long}),
        "maximum is 60s",
    )

    five_speakers = "\n".join(f"Speaker {index}: hello" for index in range(5))
    _assert_error(
        _post(
            omni_server,
            {
                **valid,
                "input": five_speakers,
                "ref_audio": [_REFERENCE_DATA_URL] * 5,
            },
        ),
        "at most 4",
    )
