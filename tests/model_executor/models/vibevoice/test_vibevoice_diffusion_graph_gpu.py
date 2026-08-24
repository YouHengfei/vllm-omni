# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU acceptance for the Phase C1 diffusion-loop CUDA-graph executor.

Replay must be bitwise identical to the eager loop for the same inputs,
across consecutive tokens (proving no cross-replay state leakage) and across
active-batch sizes. Capture failure must fall back to eager permanently.
"""

from __future__ import annotations

import os

import pytest
import torch

from vllm_omni.model_executor.models.vibevoice.diffusion import (
    VibeVoiceDiffusionGraphExecutor,
    VibeVoiceDiffusionHead,
    VibeVoiceDiffusionSampler,
)
from vllm_omni.transformers_utils.configs.vibevoice import VibeVoiceConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]

_MODEL = os.getenv("VIBEVOICE_TEST_MODEL", "microsoft/VibeVoice-1.5B")


def _build() -> tuple[VibeVoiceDiffusionHead, VibeVoiceDiffusionSampler]:
    config = VibeVoiceConfig.from_pretrained(_MODEL)
    torch.manual_seed(0)
    head = VibeVoiceDiffusionHead(config).to(device="cuda", dtype=torch.bfloat16).eval()
    sampler = VibeVoiceDiffusionSampler.from_model_config(config)
    return head, sampler


def _inputs(batch: int, hidden: int, latent: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    positive = torch.randn(batch, hidden, device="cuda", dtype=torch.bfloat16, generator=generator)
    negative = torch.randn(batch, hidden, device="cuda", dtype=torch.bfloat16, generator=generator)
    noise = torch.randn(2 * batch, latent, device="cuda", dtype=torch.bfloat16, generator=generator)
    return positive, negative, noise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_diffusion_graph_replay_is_bitwise_identical_across_tokens(batch_size: int) -> None:
    head, sampler = _build()
    executor = VibeVoiceDiffusionGraphExecutor(sampler, head)
    hidden = sampler.condition_size
    latent = sampler.latent_size

    with torch.inference_mode():
        for token_index in range(3):
            positive, negative, noise = _inputs(batch_size, hidden, latent, seed=100 + token_index)
            expected = sampler.sample_audio_latent(
                head,
                positive,
                negative,
                noise.clone(),
                guidance_scale=1.3,
                num_inference_steps=10,
            )
            actual = executor.sample(
                positive,
                negative,
                noise.clone(),
                guidance_scale=1.3,
                num_inference_steps=10,
            )
            assert actual is not None
            assert torch.equal(actual, expected), (
                f"token {token_index}: graph replay diverged from eager "
                f"(max diff {(actual.float() - expected.float()).abs().max().item()})"
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_diffusion_graph_cache_is_bounded_to_official_control_keys() -> None:
    head, sampler = _build()
    executor = VibeVoiceDiffusionGraphExecutor(sampler, head)
    hidden = sampler.condition_size
    latent = sampler.latent_size

    with torch.inference_mode():
        for batch_size in (1, 2, 3, 4):
            positive, negative, noise = _inputs(batch_size, hidden, latent, seed=7 + batch_size)
            expected = sampler.sample_audio_latent(
                head,
                positive,
                negative,
                noise.clone(),
                guidance_scale=1.3,
                num_inference_steps=10,
            )
            actual = executor.sample(
                positive,
                negative,
                noise.clone(),
                guidance_scale=1.3,
                num_inference_steps=10,
            )
            assert actual is not None
            assert torch.equal(actual, expected)

        assert executor.num_captured_graphs == 4

        positive, negative, noise = _inputs(2, hidden, latent, seed=17)
        for guidance, steps in ((1.0, 10), (1.3, 5), (2.5, 7), (1.3, 50)):
            assert (
                executor.sample(
                    positive,
                    negative,
                    noise.clone(),
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                )
                is None
            )
            assert executor.num_captured_graphs == 4

        positive5, negative5, noise5 = _inputs(5, hidden, latent, seed=18)
        assert (
            executor.sample(
                positive5,
                negative5,
                noise5,
                guidance_scale=1.3,
                num_inference_steps=10,
            )
            is None
        )
        assert executor.num_captured_graphs == 4

        expected_replay = sampler.sample_audio_latent(
            head,
            positive,
            negative,
            noise.clone(),
            guidance_scale=1.3,
            num_inference_steps=10,
        )
        replayed = executor.sample(
            positive,
            negative,
            noise.clone(),
            guidance_scale=1.3,
            num_inference_steps=10,
        )
        assert replayed is not None
        assert torch.equal(replayed, expected_replay)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_diffusion_graph_capture_failure_falls_back_to_eager() -> None:
    _, sampler = _build()

    class _BrokenHead:
        def parameters(self):
            return iter([])

    executor = VibeVoiceDiffusionGraphExecutor(sampler, _BrokenHead())
    hidden = sampler.condition_size
    latent = sampler.latent_size
    positive, negative, noise = _inputs(1, hidden, latent, seed=11)

    with torch.inference_mode():
        assert executor.sample(positive, negative, noise, guidance_scale=1.3, num_inference_steps=10) is None
        # Permanently disabled: a working input later still returns None.
        assert executor.sample(positive, negative, noise, guidance_scale=1.3, num_inference_steps=10) is None
        assert executor._disabled is True

        strict_executor = VibeVoiceDiffusionGraphExecutor(
            sampler,
            _BrokenHead(),
            capture_failure_fatal=True,
        )
        with pytest.raises(RuntimeError, match="Required VibeVoice diffusion CUDA-graph capture failed"):
            strict_executor.sample(
                positive,
                negative,
                noise,
                guidance_scale=1.3,
                num_inference_steps=10,
            )
        with pytest.raises(RuntimeError, match="disabled after a prior capture failure"):
            strict_executor.sample(
                positive,
                negative,
                noise,
                guidance_scale=1.3,
                num_inference_steps=10,
            )
