# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU acceptance: conv cache resets at segment boundaries (no cross-speaker leak).

Validates the correctness fix that zeros the causal Conv1d padding caches at
every ``audio_bos``. After a segment runs (cache accumulates left-context),
resetting the cache must make the next segment's first token bitwise
identical to decoding that token from a fresh (uninitialized) cache.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

_MODEL_ENV = "VIBEVOICE_TEST_MODEL"


def _build():
    import os

    from transformers import AutoModel

    from vllm_omni.model_executor.models.vibevoice.audio_decode import (
        VibeVoiceAudioTokenDecoder,
    )
    from vllm_omni.model_executor.models.vibevoice.vibevoice import (
        VibeVoiceMultiModalProjector,
    )
    from vllm_omni.transformers_utils.configs.vibevoice import VibeVoiceConfig

    model = os.getenv(_MODEL_ENV, "microsoft/VibeVoice-1.5B")
    config = VibeVoiceConfig.from_pretrained(model)
    torch.manual_seed(0)
    audio_tower = AutoModel.from_config(config.audio_config).to(device="cuda", dtype=torch.bfloat16).eval()
    semantic_encoder = (
        AutoModel.from_config(config.semantic_model_config).to(device="cuda", dtype=torch.bfloat16).eval()
    )
    acoustic_projector = (
        VibeVoiceMultiModalProjector(config.audio_config.hidden_size, config.hidden_size)
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    semantic_connector = (
        VibeVoiceMultiModalProjector(config.semantic_model_config.hidden_size, config.hidden_size)
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    latent_scaling = torch.tensor(1.0, device="cuda", dtype=torch.bfloat16)
    latent_bias = torch.tensor(0.0, device="cuda", dtype=torch.bfloat16)
    decoder = VibeVoiceAudioTokenDecoder.from_model_config(config)
    return decoder, audio_tower, semantic_encoder, acoustic_projector, semantic_connector, latent_scaling, latent_bias


def _decode(decoder, at, se, ap, sc, ls, lb, latent, ac, sec):
    return decoder.decode_audio_token(
        audio_tower=at,
        semantic_encoder=se,
        acoustic_projector=ap,
        semantic_connector=sc,
        latent_scaling_factor=ls,
        latent_bias_factor=lb,
        audio_latent=latent,
        acoustic_cache=ac,
        semantic_cache=sec,
    )


def _reset_cache(cache):
    for layer in cache.layers.values():
        if getattr(layer, "is_initialized", False) and layer.cache is not None:
            layer.cache.zero_()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_conv_cache_reset_makes_segment2_bitwise_equal_to_fresh_start() -> None:
    """After segment 1 accumulates cache, reset must yield a fresh segment 2."""
    decoder, at, se, ap, sc, ls, lb = _build()
    latent_size = decoder.latent_size

    with torch.inference_mode():
        # Segment 1: 5 tokens, cache accumulates.
        ac = sec = None
        for _ in range(5):
            latent = torch.randn(1, 1, latent_size, device="cuda", dtype=torch.bfloat16)
            out = _decode(decoder, at, se, ap, sc, ls, lb, latent, ac, sec)
            ac, sec = out.acoustic_cache, out.semantic_cache

        # Reset caches (simulates _start_audio_segment at audio_bos).
        _reset_cache(ac)
        _reset_cache(sec)

        # Segment 2 token 1 with the reset (zeroed) caches.
        seg2_latent = torch.randn(1, 1, latent_size, device="cuda", dtype=torch.bfloat16)
        reset_out = _decode(decoder, at, se, ap, sc, ls, lb, seg2_latent, ac, sec)

        # Oracle: decode the same latent from a completely fresh (None) cache.
        fresh_out = _decode(decoder, at, se, ap, sc, ls, lb, seg2_latent, None, None)

    assert torch.equal(reset_out.audio, fresh_out.audio), (
        "segment-2 audio after cache reset differs from a fresh start "
        f"(max diff {(reset_out.audio.float() - fresh_out.audio.float()).abs().max().item()})"
    )
    assert torch.equal(reset_out.next_embedding, fresh_out.next_embedding), (
        "segment-2 embedding after cache reset differs from a fresh start "
        f"(max diff {(reset_out.next_embedding.float() - fresh_out.next_embedding.float()).abs().max().item()})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_conv_cache_without_reset_leaks_into_segment2() -> None:
    """Sanity check: WITHOUT reset, segment 2 differs from a fresh start.

    This confirms the test above is meaningful (the leak is real when reset
    is skipped) and documents the pre-fix behavior inherited from the PR.
    """
    decoder, at, se, ap, sc, ls, lb = _build()
    latent_size = decoder.latent_size

    with torch.inference_mode():
        ac = sec = None
        for _ in range(5):
            latent = torch.randn(1, 1, latent_size, device="cuda", dtype=torch.bfloat16)
            out = _decode(decoder, at, se, ap, sc, ls, lb, latent, ac, sec)
            ac, sec = out.acoustic_cache, out.semantic_cache

        seg2_latent = torch.randn(1, 1, latent_size, device="cuda", dtype=torch.bfloat16)
        leaked_out = _decode(decoder, at, se, ap, sc, ls, lb, seg2_latent, ac, sec)
        fresh_out = _decode(decoder, at, se, ap, sc, ls, lb, seg2_latent, None, None)

    assert not torch.equal(leaked_out.audio, fresh_out.audio), (
        "Without reset, segment 2 unexpectedly matched a fresh start (the leak should produce different output)."
    )
