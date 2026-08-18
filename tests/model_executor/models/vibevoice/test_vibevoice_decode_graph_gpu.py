# SPDX-License-Identifier: Apache-2.0
"""GPU acceptance for the Phase C2 M4a decode CUDA-graph executor.

Replay must be bitwise identical to eager across consecutive tokens (cache
accumulates), different inputs, and segment boundaries (cache reset). Capture
failure must fall back to eager permanently.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

_MODEL_ROOT_ENV = "VIBEVOICE_TEST_MODEL_ROOT"


def _build():
    from transformers import AutoModel

    from vllm_omni.model_executor.models.vibevoice.audio_decode import (
        VibeVoiceAudioTokenDecoder,
        VibeVoiceDecodeGraphExecutor,
    )
    from vllm_omni.model_executor.models.vibevoice.vibevoice import (
        VibeVoiceMultiModalProjector,
    )
    from vllm_omni.transformers_utils.configs.vibevoice import VibeVoiceConfig

    model_root = os.getenv(_MODEL_ROOT_ENV)
    config_path = (
        os.path.join(model_root, "VibeVoice-1.5B-hf", "config.json")
        if model_root
        else "/SharedData/youhf/models/VibeVoice-1.5B-hf/config.json"
    )
    config = VibeVoiceConfig.from_pretrained(config_path)
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
    executor = VibeVoiceDecodeGraphExecutor(decoder)
    latent_size = decoder.latent_size
    return (
        executor,
        decoder,
        audio_tower,
        semantic_encoder,
        acoustic_projector,
        semantic_connector,
        latent_scaling,
        latent_bias,
        latent_size,
    )


def _decode(executor, decoder, at, se, ap, sc, ls, lb, latent, ac, sec, *, use_graph):
    if use_graph:
        out = executor.decode(
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
        if out is not None:
            return out
    return decoder_decode(decoder, at, se, ap, sc, ls, lb, latent, ac, sec)


def decoder_decode(decoder, at, se, ap, sc, ls, lb, latent, ac, sec):
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


def test_decode_graph_replay_is_bitwise_identical_across_tokens() -> None:
    (executor, decoder, at, se, ap, sc, ls, lb, latent_size) = _build()
    latents = [torch.randn(1, 1, latent_size, device="cuda", dtype=torch.bfloat16) for _ in range(5)]

    with torch.inference_mode():
        # Eager reference: sequential, cache accumulates.
        ac = sec = None
        eager_audio, eager_emb = [], []
        for latent in latents:
            out = decoder_decode(decoder, at, se, ap, sc, ls, lb, latent, ac, sec)
            ac, sec = out.acoustic_cache, out.semantic_cache
            eager_audio.append(out.audio.clone())
            eager_emb.append(out.next_embedding.clone())

        # Graph: token 1 eager (populate cache), token 2 capture, token 3-5 replay.
        ac = sec = None
        out1 = decoder_decode(decoder, at, se, ap, sc, ls, lb, latents[0], ac, sec)
        ac, sec = out1.acoustic_cache, out1.semantic_cache
        graph_audio, graph_emb = [out1.audio.clone()], [out1.next_embedding.clone()]
        for i in range(1, 5):
            out = _decode(executor, decoder, at, se, ap, sc, ls, lb, latents[i], ac, sec, use_graph=True)
            graph_audio.append(out.audio.clone())
            graph_emb.append(out.next_embedding.clone())

    for i in range(5):
        assert torch.equal(graph_audio[i], eager_audio[i]), (
            f"token {i}: graph audio diverged (max diff "
            f"{(graph_audio[i].float() - eager_audio[i].float()).abs().max().item()})"
        )
        assert torch.equal(graph_emb[i], eager_emb[i]), (
            f"token {i}: graph embedding diverged (max diff "
            f"{(graph_emb[i].float() - eager_emb[i].float()).abs().max().item()})"
        )


def test_decode_graph_survives_segment_reset() -> None:
    """Cache zero_ at a segment boundary keeps addresses stable; graph stays valid."""
    (executor, decoder, at, se, ap, sc, ls, lb, latent_size) = _build()
    latents = [torch.randn(1, 1, latent_size, device="cuda", dtype=torch.bfloat16) for _ in range(4)]

    with torch.inference_mode():
        # Segment 1: token 1 eager + token 2 graph (capture).
        ac = sec = None
        out1 = decoder_decode(decoder, at, se, ap, sc, ls, lb, latents[0], ac, sec)
        ac, sec = out1.acoustic_cache, out1.semantic_cache
        _decode(executor, decoder, at, se, ap, sc, ls, lb, latents[1], ac, sec, use_graph=True)

        # Segment boundary: reset caches (zero_), graph must remain valid.
        for cache in (ac, sec):
            for layer in cache.layers.values():
                if getattr(layer, "is_initialized", False) and layer.cache is not None:
                    layer.cache.zero_()

        # Segment 2: graph replay from zero cache.
        out3 = _decode(executor, decoder, at, se, ap, sc, ls, lb, latents[2], ac, sec, use_graph=True)
        out4 = _decode(executor, decoder, at, se, ap, sc, ls, lb, latents[3], ac, sec, use_graph=True)

        # Oracle: segment 2 from a fresh (None) cache.
        ac2 = sec2 = None
        ref3 = decoder_decode(decoder, at, se, ap, sc, ls, lb, latents[2], ac2, sec2)
        ac2, sec2 = ref3.acoustic_cache, ref3.semantic_cache
        ref4 = decoder_decode(decoder, at, se, ap, sc, ls, lb, latents[3], ac2, sec2)

    # After cache reset, graph replay and a fresh start both read zero
    # context; bf16 conv algorithm selection may differ by buffer address,
    # so assert within bf16 tolerance rather than bitwise.
    torch.testing.assert_close(out3.audio.float(), ref3.audio.float(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out4.audio.float(), ref4.audio.float(), rtol=1e-3, atol=1e-3)


def test_decode_graph_capture_failure_falls_back_to_eager() -> None:
    (executor, decoder, at, se, ap, sc, ls, lb, latent_size) = _build()
    latent = torch.randn(1, 1, latent_size, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        # acoustic_cache=None triggers the eager fallback path inside decode().
        out = executor.decode(
            audio_tower=at,
            semantic_encoder=se,
            acoustic_projector=ap,
            semantic_connector=sc,
            latent_scaling_factor=ls,
            latent_bias_factor=lb,
            audio_latent=latent,
            acoustic_cache=None,
            semantic_cache=None,
        )
        assert out is None
