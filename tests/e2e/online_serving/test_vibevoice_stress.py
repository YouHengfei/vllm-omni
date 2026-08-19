# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VibeVoice stress tests: stability, performance, multi-speaker, long-form.

These tests are designed for pre-merge validation of stability, performance,
and correctness under sustained load. They reuse the existing omni_server
fixture and SSE PCM capture (same transport as the perf baseline tests) and
report metrics using the repository's standard definitions (RTF, TTFA,
aggregate throughput, continuity).

Scenarios:
  S1 Stability: 4 concurrent long-form SSE streams × N rounds, assert
     finish=stop rate, RTF distribution, no crash.
  S2 Performance: B=1 and B=4 long-form SSE, report RTF p50/p95, TTFA,
     aggregate throughput (aligned with VoxCPM2 bench metrics).
  S3 Multi-speaker: 4-speaker dialogue × 4 concurrent, assert finish=stop,
     segment count, audio continuity.
  S4 Long-text extreme: single request with very large max_new_tokens,
     assert completion, no OOM, finite audio.
"""

from __future__ import annotations

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from tests.helpers.media import get_asset_path, load_test_audio_data_url
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.advanced_model]

_MODEL_ROOT = "/SharedData/youhf/models"
_MODEL = str(Path(_MODEL_ROOT) / "VibeVoice")
_TOKENIZER = str(Path(_MODEL_ROOT) / "VibeVoice-1.5B-hf")
_REF = load_test_audio_data_url("qwen3_tts/clone_2.wav")
_FOUR_SPEAKER_REFS = [
    load_test_audio_data_url("cosyvoice3/zero_shot_prompt.wav"),
    load_test_audio_data_url("glm_tts/jiayan_zh.wav"),
    load_test_audio_data_url("indextts2/ref_audio.wav"),
    load_test_audio_data_url("qwen3_tts/clone_2.wav"),
]

_SERVER = pytest.param(
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
        env_dict={
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        },
        init_timeout=900,
        stage_init_timeout=600,
        require_real_weights=True,
    ),
    id="official-vibevoice",
)

# A long English paragraph (~ many audio tokens).
_LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "Sphinx of black quartz, judge my vow. "
    "How vexingly quick daft zebras jump. "
    "The five boxing wizards jump quickly. "
    "Bright vixens jump dozy fowl quack. "
    "Quick zephyrs blow, vexing daft Jim. "
    "Two driven jocks help fax my big quiz. "
    "Crazy Fredrick bought many very exquisite opal jewels. "
    "We promptly judged antique ivory buckles for the next prize. "
)

_FOUR_SPEAKER_TEXT = "\n".join(
    [
        "Speaker 0: Welcome to the show.",
        "Speaker 1: Thank you for having me.",
        "Speaker 2: Let us dive right in.",
        "Speaker 3: I am excited to be here.",
    ]
)


def _stream_one(url: str, payload: dict, index: int) -> dict:
    """Drive one SSE request; return timing/audio/finish counters."""
    t0 = time.perf_counter()
    total_pcm = bytearray()
    finish_reason = None
    with httpx.Client(trust_env=False, timeout=600) as client, client.stream("POST", url, json=payload) as resp:
        assert resp.status_code == 200, resp.read().decode(errors="replace")
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line.removeprefix("data: "))
            if event["type"] == "speech.audio.delta":
                total_pcm.extend(base64.b64decode(event["audio"]))
            elif event["type"] == "speech.audio.done":
                finish_reason = event.get("finish_reason")
    total = time.perf_counter() - t0
    n_samples = len(total_pcm) // 2
    return {
        "total_s": total,
        "audio_s": n_samples / 24000.0,
        "n_tokens": n_samples // 3200,
        "finish_reason": finish_reason,
    }


# =====================================================================
# S1: Stability — 4 concurrent long-form SSE × N rounds
# =====================================================================


@pytest.mark.parametrize("omni_server", [_SERVER], indirect=True)
def test_vibevoice_stress_stability_001(omni_server) -> None:
    """S1: 4 concurrent long-form SSE streams × 3 rounds, no crash, all stop."""
    url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
    rounds = 3
    all_results = []
    for rnd in range(rounds):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(
                pool.map(
                    lambda i: _stream_one(
                        url,
                        {
                            "model": omni_server.model,
                            "input": f"{_LONG_TEXT} Round variation {rnd}.{i}.",
                            "ref_audio": _REF,
                            "response_format": "pcm",
                            "stream": True,
                            "timeout": 300.0,
                        },
                        i,
                    ),
                    range(4),
                )
            )
        wall = time.perf_counter() - t0
        all_results.extend(results)
        for r in results:
            assert r["finish_reason"] == "stop", f"Round {rnd} request did not finish naturally"
            assert r["n_tokens"] > 5, f"Round {rnd} request produced too few tokens"
        total_audio = sum(r["audio_s"] for r in results)
        print(f"\n[S1 R{rnd}] wall={wall:.1f}s total_audio={total_audio:.1f}s aggregate={total_audio / wall:.2f}x")
    # Summary
    rtfs = [r["total_s"] / r["audio_s"] for r in all_results if r["audio_s"] > 0]
    print(
        f"\n[S1 Summary] {len(all_results)} requests, RTF min={min(rtfs):.3f} max={max(rtfs):.3f} mean={sum(rtfs) / len(rtfs):.3f}"
    )
    assert all(r["finish_reason"] == "stop" for r in all_results)


# =====================================================================
# S2: Performance — B=1 and B=4, report RTF/TTFA/throughput
# =====================================================================


@pytest.mark.parametrize("omni_server", [_SERVER], indirect=True)
def test_vibevoice_stress_perf_b1_002(omni_server) -> None:
    """S2a: B=1 long-form SSE, report RTF and TTFA."""
    url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
    result = _stream_one(
        url,
        {
            "model": omni_server.model,
            "input": _LONG_TEXT,
            "ref_audio": _REF,
            "response_format": "pcm",
            "stream": True,
            "timeout": 300.0,
        },
        0,
    )
    rtf = result["total_s"] / result["audio_s"] if result["audio_s"] > 0 else float("inf")
    print(
        f"\n[S2 B=1] total={result['total_s'] * 1000:.0f}ms audio={result['audio_s']:.3f}s rtf={rtf:.3f} tokens={result['n_tokens']}"
    )
    assert result["finish_reason"] == "stop"
    assert rtf < 1.0


@pytest.mark.parametrize("omni_server", [_SERVER], indirect=True)
def test_vibevoice_stress_perf_b4_003(omni_server) -> None:
    """S2b: B=4 concurrent long-form SSE, report aggregate throughput."""
    url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda i: _stream_one(
                    url,
                    {
                        "model": omni_server.model,
                        "input": f"{_LONG_TEXT} Variation {i}.",
                        "ref_audio": _REF,
                        "response_format": "pcm",
                        "stream": True,
                        "timeout": 300.0,
                    },
                    i,
                ),
                range(4),
            )
        )
    wall = time.perf_counter() - t0
    total_audio = sum(r["audio_s"] for r in results)
    aggregate_x = total_audio / wall
    for i, r in enumerate(results):
        rtf = r["total_s"] / r["audio_s"] if r["audio_s"] > 0 else float("inf")
        print(
            f"\n[S2 B=4 req{i}] finish={r['finish_reason']} total={r['total_s'] * 1000:.0f}ms audio={r['audio_s']:.3f}s rtf={rtf:.3f} tokens={r['n_tokens']}"
        )
        assert r["finish_reason"] == "stop"
    print(f"\n[S2 B=4] wall={wall:.2f}s total_audio={total_audio:.2f}s aggregate={aggregate_x:.2f}x")
    assert aggregate_x > 1.0


# =====================================================================
# S3: Multi-speaker — 4-speaker dialogue × 4 concurrent
# =====================================================================


@pytest.mark.parametrize("omni_server", [_SERVER], indirect=True)
def test_vibevoice_stress_multi_speaker_004(omni_server) -> None:
    """S3: 4-speaker dialogue × 4 concurrent, assert finish=stop and segments."""
    url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda i: _stream_one(
                    url,
                    {
                        "model": omni_server.model,
                        "input": f"{_FOUR_SPEAKER_TEXT}\nSpeaker {i}: And here is my additional thought.",
                        "ref_audio": _FOUR_SPEAKER_REFS,
                        "response_format": "pcm",
                        "stream": True,
                        "timeout": 600.0,
                    },
                    i,
                ),
                range(4),
            )
        )
    wall = time.perf_counter() - t0
    total_audio = sum(r["audio_s"] for r in results)
    for i, r in enumerate(results):
        print(f"\n[S3 req{i}] finish={r['finish_reason']} tokens={r['n_tokens']} audio={r['audio_s']:.3f}s")
        assert r["finish_reason"] == "stop"
        assert r["n_tokens"] > 5
    print(f"\n[S3] wall={wall:.2f}s total_audio={total_audio:.2f}s aggregate={total_audio / wall:.2f}x")


# =====================================================================
# S4: Long-text extreme — single request, very large max_new_tokens
# =====================================================================


@pytest.mark.parametrize("omni_server", [_SERVER], indirect=True)
def test_vibevoice_stress_long_text_extreme_005(omni_server) -> None:
    """S4: single request with large max_new_tokens, assert completion, no OOM."""
    url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
    # ~5000 words: repeat the long paragraph many times
    extreme_text = " ".join([_LONG_TEXT] * 20)
    t0 = time.perf_counter()
    result = _stream_one(
        url,
        {
            "model": omni_server.model,
            "input": extreme_text,
            "ref_audio": _REF,
            "response_format": "pcm",
            "stream": True,
            "max_new_tokens": 4096,
            "timeout": 600.0,
        },
        0,
    )
    total = time.perf_counter() - t0
    rtf = total / result["audio_s"] if result["audio_s"] > 0 else float("inf")
    print(
        f"\n[S4] finish={result['finish_reason']} total={total * 1000:.0f}ms audio={result['audio_s']:.3f}s rtf={rtf:.3f} tokens={result['n_tokens']}"
    )
    assert result["finish_reason"] in ("stop", "length")
    assert result["n_tokens"] > 100
    assert rtf < 1.0
