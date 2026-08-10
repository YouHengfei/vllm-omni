# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the model-local VibeVoice diffusion numerical kernel."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.vibevoice.diffusion import (
    VibeVoiceDiffusionSampler,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DeterministicDiffusionHead(nn.Module):
    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        latent_size = noisy_latents.shape[-1]
        return (
            noisy_latents * 0.125
            + condition[:, :latent_size] * 0.0625
            + timesteps[:, None] * 1e-4
        )


def _sampler() -> VibeVoiceDiffusionSampler:
    config = SimpleNamespace(
        ddpm_num_steps=1_000,
        ddpm_num_inference_steps=10,
        ddpm_beta_schedule="cosine",
        prediction_type="v_prediction",
        hidden_size=96,
        audio_config=SimpleNamespace(hidden_size=64),
    )
    return VibeVoiceDiffusionSampler.from_model_config(config)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive = torch.linspace(-0.5, 0.5, 2 * 96).reshape(2, 96)
    negative = torch.linspace(0.25, -0.25, 2 * 96).reshape(2, 96)
    noise = torch.linspace(-1.0, 1.0, 4 * 64).reshape(4, 64)
    return positive, negative, noise


def _reference_sample(
    scheduler,
    head: nn.Module,
    positive: torch.Tensor,
    negative: torch.Tensor,
    noise: torch.Tensor,
    *,
    guidance_scale: float,
    num_inference_steps: int,
) -> torch.Tensor:
    batch_size = positive.shape[0]
    condition = torch.cat([positive, negative], dim=0)
    latent = noise.to(condition).clone()
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    for timestep in scheduler.timesteps:
        combined = torch.cat([latent[:batch_size], latent[:batch_size]], dim=0)
        prediction = head(
            combined,
            timestep.repeat(combined.shape[0]).to(combined),
            condition,
        )
        conditional = prediction[:batch_size]
        unconditional = prediction[batch_size:]
        guided = unconditional + guidance_scale * (conditional - unconditional)
        latent = scheduler.step(
            torch.cat([guided, guided], dim=0),
            timestep,
            latent,
        ).prev_sample
    return latent[:batch_size].unsqueeze(1)


def _load_microsoft_scheduler():
    official_repo = os.getenv("VIBEVOICE_OFFICIAL_REPO")
    if not official_repo:
        pytest.skip("Set VIBEVOICE_OFFICIAL_REPO for Microsoft DPM solver parity")
    scheduler_path = Path(official_repo) / "vibevoice/schedule/dpm_solver.py"
    if not scheduler_path.is_file():
        pytest.skip(f"Microsoft DPM solver not found at {scheduler_path}")

    spec = importlib.util.spec_from_file_location(
        "_vibevoice_official_dpm_solver",
        scheduler_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DPMSolverMultistepScheduler


def test_diffusion_sampler_builds_normalized_fresh_schedulers() -> None:
    sampler = _sampler()
    assert sampler.beta_schedule == "squaredcos_cap_v2"
    assert sampler.prediction_type == "v_prediction"
    assert sampler.condition_size == 96
    assert sampler.latent_size == 64
    assert sampler.default_num_inference_steps == 10

    first = sampler.create_scheduler()
    second = sampler.create_scheduler()
    assert first is not second
    first.set_timesteps(num_inference_steps=10)
    assert first.num_inference_steps == 10
    assert second.num_inference_steps is None
    assert second.step_index is None


def test_diffusion_kernel_matches_an_independent_reference_loop() -> None:
    sampler = _sampler()
    head = _DeterministicDiffusionHead()
    positive, negative, noise = _inputs()
    original_noise = noise.clone()

    actual = sampler.sample_audio_latent(
        head,
        positive,
        negative,
        noise,
        guidance_scale=1.3,
        num_inference_steps=10,
    )
    expected = _reference_sample(
        sampler.create_scheduler(),
        head,
        positive,
        negative,
        noise,
        guidance_scale=1.3,
        num_inference_steps=10,
    )

    assert actual.shape == (2, 1, 64)
    assert torch.equal(actual, expected)
    assert torch.equal(noise, original_noise)
    assert torch.isfinite(actual).all()


def test_diffusers_scheduler_is_step_exact_with_microsoft_solver() -> None:
    microsoft_scheduler_cls = _load_microsoft_scheduler()
    sampler = _sampler()
    head = _DeterministicDiffusionHead()
    positive, negative, noise = _inputs()

    microsoft_scheduler = microsoft_scheduler_cls(
        num_train_timesteps=sampler.num_train_timesteps,
        beta_schedule="cosine",
        prediction_type=sampler.prediction_type,
    )
    runtime_scheduler = sampler.create_scheduler()
    microsoft_scheduler.set_timesteps(num_inference_steps=10)
    runtime_scheduler.set_timesteps(num_inference_steps=10)

    assert torch.equal(microsoft_scheduler.betas, runtime_scheduler.betas)
    assert torch.equal(microsoft_scheduler.timesteps, runtime_scheduler.timesteps)

    microsoft_latent = noise.clone()
    runtime_latent = noise.clone()
    condition = torch.cat([positive, negative], dim=0)
    batch_size = positive.shape[0]
    for microsoft_timestep, runtime_timestep in zip(
        microsoft_scheduler.timesteps,
        runtime_scheduler.timesteps,
        strict=True,
    ):
        combined = torch.cat(
            [microsoft_latent[:batch_size], microsoft_latent[:batch_size]],
            dim=0,
        )
        prediction = head(
            combined,
            microsoft_timestep.repeat(combined.shape[0]).to(combined),
            condition,
        )
        conditional = prediction[:batch_size]
        unconditional = prediction[batch_size:]
        guided = unconditional + 1.3 * (conditional - unconditional)
        solver_prediction = torch.cat([guided, guided], dim=0)

        microsoft_latent = microsoft_scheduler.step(
            solver_prediction,
            microsoft_timestep,
            microsoft_latent,
        ).prev_sample
        runtime_latent = runtime_scheduler.step(
            solver_prediction,
            runtime_timestep,
            runtime_latent,
        ).prev_sample
        assert torch.equal(microsoft_latent, runtime_latent)

    microsoft_result = _reference_sample(
        microsoft_scheduler_cls(
            num_train_timesteps=sampler.num_train_timesteps,
            beta_schedule="cosine",
            prediction_type=sampler.prediction_type,
        ),
        head,
        positive,
        negative,
        noise,
        guidance_scale=1.3,
        num_inference_steps=10,
    )
    runtime_result = sampler.sample_audio_latent(
        head,
        positive,
        negative,
        noise,
        guidance_scale=1.3,
        num_inference_steps=10,
    )
    assert torch.equal(microsoft_result, runtime_result)


@pytest.mark.parametrize(
    ("positive", "negative", "noise", "guidance_scale", "steps", "message"),
    [
        (
            torch.zeros(2, 96),
            torch.zeros(1, 96),
            torch.zeros(4, 64),
            1.3,
            10,
            "condition shapes must match",
        ),
        (
            torch.zeros(2, 96),
            torch.zeros(2, 96),
            torch.zeros(2, 64),
            1.3,
            10,
            "noise must preserve the official cond/uncond shape",
        ),
        (
            torch.zeros(2, 96),
            torch.zeros(2, 96),
            torch.zeros(4, 64),
            float("nan"),
            10,
            "guidance_scale must be finite",
        ),
        (
            torch.zeros(2, 96),
            torch.zeros(2, 96),
            torch.zeros(4, 64),
            1.3,
            0,
            "num_inference_steps must be positive",
        ),
    ],
)
def test_diffusion_kernel_rejects_invalid_contracts(
    positive: torch.Tensor,
    negative: torch.Tensor,
    noise: torch.Tensor,
    guidance_scale: float,
    steps: int,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _sampler().sample_audio_latent(
            _DeterministicDiffusionHead(),
            positive,
            negative,
            noise,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
        )
