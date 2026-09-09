# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online serving E2E for VibeVoice TTS on the support-tecoomni (Plan-B) baseline.

Starts the OpenAI-compatible server with a real VibeVoice checkpoint and POSTs
/v1/audio/speech with a bundled reference voice, asserting a well-formed 24 kHz
WAV response. This exercises the serving adapter (prompt rendering, reference
audio resolution, token gating) and the model's full flow over HTTP.

Environment variables:
    VIBEVOICE_TEST_MODEL     checkpoint dir (default /SharedData/youhf/models/VibeVoice)
    VIBEVOICE_TEST_TOKENIZER tokenizer repo (default Qwen/Qwen2.5-1.5B)

Note: the test process and the server subprocess both set NO_PROXY so that
localhost requests bypass any developer/CI HTTP proxy (a proxy that intercepts
127.0.0.1 traffic otherwise returns empty 503s).
"""

from __future__ import annotations

import base64
import io
import os
import time

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.tts]

_MODEL = os.getenv("VIBEVOICE_TEST_MODEL", "/SharedData/youhf/models/VibeVoice")
_TOKENIZER = os.getenv("VIBEVOICE_TEST_TOKENIZER", "Qwen/Qwen2.5-1.5B")
_DEPLOY = os.getenv("VIBEVOICE_TEST_DEPLOY_CONFIG", "/tmp/vibevoice_e2e.yaml")
_REF = os.path.join(
    os.path.dirname(__file__),
    "../../../vllm_omni/model_executor/models/vibevoice/assets/default_0.wav",
)

# Localhost traffic must bypass the ambient proxy, else requests 503.
_NO_PROXY = {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}


def _wait_ready(base: str, timeout_s: float = 600.0) -> None:
    import httpx

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            if httpx.get(f"{base}/health", timeout=5.0).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("server did not become healthy in time")


@pytest.mark.skipif(not os.path.isdir(_MODEL), reason=f"VibeVoice checkpoint not found at {_MODEL}")
def test_vibevoice_online_speech() -> None:
    import torch as _torch

    if not _torch.cuda.is_available():
        pytest.skip("requires a GPU")
    os.environ.update(_NO_PROXY)

    import httpx
    import soundfile as sf

    from tests.helpers.runtime import OmniServer

    env = {
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        **_NO_PROXY,
    }
    serve_args = [
        "--stage-configs-path",
        _DEPLOY,
        "--tokenizer",
        _TOKENIZER,
        "--disable-log-stats",
    ]
    with OmniServer(_MODEL, serve_args, env_dict=env) as server:
        base = f"http://{server.host}:{server.port}"
        _wait_ready(base)

        waveform, sample_rate = sf.read(_REF, dtype="float32")
        buffer = io.BytesIO()
        sf.write(buffer, waveform, sample_rate, format="WAV")
        data_url = "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        payload = {
            "model": _MODEL,
            "input": "Hello, this is an online VibeVoice serving test.",
            "ref_audio": data_url,
            "response_format": "wav",
        }
        response = httpx.post(f"{base}/v1/audio/speech", json=payload, timeout=300.0)
        assert response.status_code == 200, response.text[:500]

        audio, out_sr = sf.read(io.BytesIO(response.content), dtype="float32")
        assert out_sr == 24000
        assert audio.ndim == 1 and audio.shape[0] > 0
        assert audio.shape[0] % 3200 == 0
        assert bool(np.isfinite(audio).all())
        assert float(np.abs(audio).max()) > 0.01  # not silence
