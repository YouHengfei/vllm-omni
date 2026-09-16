# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Comprehensive online serving E2E for VibeVoice TTS (support-tecoomni baseline).

Covers the production serving paths beyond the single-request smoke test:
- Batched concurrent requests (mixed prefill/decode -> exercises the
  postprocess_requires_all_scheduled_requests path).
- Multi-speaker voice cloning with the four bundled default voices.
- WebSocket streaming (/v1/audio/speech/stream).

Shares one server across the module (startup is ~2 min). Requires a GPU, the
real VibeVoice checkpoint, and the bundled default voices.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.tts]

_MODEL = os.getenv("VIBEVOICE_TEST_MODEL", "/SharedData/youhf/models/VibeVoice")
_TOKENIZER = os.getenv("VIBEVOICE_TEST_TOKENIZER", "Qwen/Qwen2.5-1.5B")
_DEPLOY = os.getenv("VIBEVOICE_TEST_DEPLOY_CONFIG", "/tmp/vibevoice_batch.yaml")
_ASSETS = os.path.join(
    os.path.dirname(__file__),
    "../../../vllm_omni/model_executor/models/vibevoice/assets",
)
_NO_PROXY = {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}


def _data_url(path: str) -> str:
    import soundfile as sf

    waveform, sample_rate = sf.read(path, dtype="float32")
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV")
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


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


@pytest.fixture(scope="module")
def omni_server():
    import torch as _torch

    if not _torch.cuda.is_available():
        pytest.skip("requires a GPU")
    if not os.path.isdir(_MODEL):
        pytest.skip(f"VibeVoice checkpoint not found at {_MODEL}")
    os.environ.update(_NO_PROXY)

    from tests.helpers.runtime import OmniServer

    env = {
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        **_NO_PROXY,
    }
    serve_args = ["--stage-configs-path", _DEPLOY, "--tokenizer", _TOKENIZER, "--disable-log-stats"]
    with OmniServer(_MODEL, serve_args, env_dict=env) as server:
        base = f"http://{server.host}:{server.port}"
        _wait_ready(base)
        yield base, f"ws://{server.host}:{server.port}"


def _assert_wav(content: bytes, label: str) -> None:
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(content), dtype="float32")
    assert sr == 24000, f"{label}: sr={sr}"
    assert audio.ndim == 1 and audio.shape[0] > 0, f"{label}: shape={audio.shape}"
    assert audio.shape[0] % 3200 == 0, f"{label}: not a whole # of audio tokens"
    assert bool(np.isfinite(audio).all()), f"{label}: non-finite"
    assert float(np.abs(audio).max()) > 0.01, f"{label}: silence"


@pytest.mark.skipif(not os.path.isdir(_MODEL), reason="no checkpoint")
def test_vibevoice_batched_concurrent(omni_server) -> None:
    """Four concurrent requests must all succeed (mixed prefill/decode batch)."""
    import concurrent.futures

    import httpx

    base, _ = omni_server
    ref = _data_url(os.path.join(_ASSETS, "default_0.wav"))
    texts = [
        "The first speaker says hello.",
        "A second voice greets you now.",
        "Third request in the batch here.",
        "And the fourth one finishes the set.",
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [
            ex.submit(
                httpx.post,
                f"{base}/v1/audio/speech",
                json={"model": _MODEL, "input": t, "ref_audio": ref, "response_format": "wav"},
                timeout=300.0,
            )
            for t in texts
        ]
        responses = [f.result() for f in futs]
    for i, r in enumerate(responses):
        assert r.status_code == 200, f"req{i}: {r.status_code} {r.text[:200]}"
        _assert_wav(r.content, f"batch{i}")


@pytest.mark.skipif(not os.path.isdir(_MODEL), reason="no checkpoint")
def test_vibevoice_multi_speaker_four_voices(omni_server) -> None:
    """Four-speaker script with the four bundled default voices."""
    import httpx

    base, _ = omni_server
    refs = [_data_url(os.path.join(_ASSETS, f"default_{i}.wav")) for i in range(4)]
    script = (
        "Speaker 0: Hello from the first voice.\n"
        "Speaker 1: And the second voice speaks.\n"
        "Speaker 2: The third voice joins in.\n"
        "Speaker 3: Fourth voice wraps it up."
    )
    r = httpx.post(
        f"{base}/v1/audio/speech",
        json={"model": _MODEL, "input": script, "ref_audio": refs, "response_format": "wav"},
        timeout=300.0,
    )
    assert r.status_code == 200, r.text[:300]
    _assert_wav(r.content, "multi4")


@pytest.mark.skipif(not os.path.isdir(_MODEL), reason="no checkpoint")
def test_vibevoice_websocket_stream(omni_server) -> None:
    """WebSocket streaming: session.config -> input.text -> input.done -> audio."""
    import asyncio

    import soundfile as sf
    import websockets

    _, ws_base = omni_server
    ref = _data_url(os.path.join(_ASSETS, "default_0.wav"))

    async def run() -> None:
        async with websockets.connect(f"{ws_base}/v1/audio/speech/stream", max_size=64 * 1024 * 1024) as ws:
            await ws.send(json.dumps({"type": "session.config", "model": _MODEL, "ref_audio": ref, "response_format": "wav"}))
            await ws.send(json.dumps({"type": "input.text", "text": "Hello, this is a streaming VibeVoice test."}))
            await ws.send(json.dumps({"type": "input.done"}))
            chunks = []
            got_start = got_session_done = False
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=300)
                if isinstance(msg, bytes):
                    chunks.append(msg)
                else:
                    data = json.loads(msg)
                    if data["type"] == "audio.start":
                        got_start = True
                    elif data["type"] == "session.done":
                        got_session_done = True
                        break
                    elif data["type"] == "error":
                        pytest.fail(f"stream error: {data.get('message')}")
            audio = b"".join(chunks)
            assert got_start and got_session_done
            assert len(audio) > 0
            wav, sr = sf.read(io.BytesIO(audio), dtype="float32")
            assert sr == 24000 and wav.ndim == 1 and bool(np.isfinite(wav).all())

    asyncio.run(run())
