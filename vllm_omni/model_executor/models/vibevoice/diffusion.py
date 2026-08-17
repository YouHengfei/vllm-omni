# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#

"""Diffusion Head and per-audio-token sampling for non-Realtime VibeVoice.

The learned Head performs one denoising prediction. The parameter-free sampler
combines repeated Head calls with CFG and a fresh DPM solver. This module owns
no request, Qwen KV-cache, decoder-cache, or scheduler-runtime state; callers
supply positive/negative Qwen conditions and the complete initial noise tensor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers.activations import ACT2FN


class VibeVoiceRMSNorm(nn.Module):
    """RMSNorm shared by the released VibeVoice projectors and head."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class VibeVoiceDiffusionHeadSinusoidalEmbedding(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        dim = config.frequency_embedding_size // 2
        freq = torch.exp(-math.log(config.diffusion_max_period) * torch.arange(dim, dtype=torch.float32) / dim)
        self.frequency_embedding_size = config.frequency_embedding_size
        self.register_buffer("freq", freq, persistent=False)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        args = timesteps[:, None].float() * self.freq[None].to(timesteps.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = nn.functional.pad(embedding, (0, 1))
        return embedding


class VibeVoiceDiffusionHeadMLP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(
            config.frequency_embedding_size,
            config.hidden_size,
            bias=False,
        )
        self.act = ACT2FN[config.hidden_act]
        self.layer_2 = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layer_2(self.act(self.layer_1(hidden_states)))


class VibeVoiceDiffusionHeadMLPBlock(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class VibeVoiceDiffusionHeadAdaLayerNorm(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.ffn = VibeVoiceDiffusionHeadMLPBlock(config)
        self.norm = VibeVoiceRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.linear = nn.Linear(
            config.hidden_size,
            config.hidden_size * 3,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale, gate = self.linear(self.act_fn(condition)).chunk(3, dim=-1)
        modulated = self.norm(hidden_states) * (1 + scale) + shift
        return hidden_states + gate * self.ffn(modulated)


class VibeVoiceDiffusionHeadFinalLayer(nn.Module):
    def __init__(self, config: Any, output_size: int) -> None:
        super().__init__()
        self.norm_eps = config.rms_norm_eps
        self.linear_1 = nn.Linear(
            config.hidden_size,
            config.hidden_size * 2,
            bias=False,
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.linear_2 = nn.Linear(config.hidden_size, output_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.linear_1(self.act_fn(condition)).chunk(2, dim=-1)
        hidden_states = hidden_states * torch.rsqrt(hidden_states.square().mean(-1, keepdim=True) + self.norm_eps)
        return self.linear_2(hidden_states * (1 + scale) + shift)


class VibeVoiceDiffusionHead(nn.Module):
    """One learned VibeVoice denoising prediction at a single timestep."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.noisy_images_proj = nn.Linear(
            config.audio_config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.cond_proj = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.timestep_embedding = VibeVoiceDiffusionHeadSinusoidalEmbedding(config)
        self.timestep_proj = VibeVoiceDiffusionHeadMLP(config)
        self.layers = nn.ModuleList(VibeVoiceDiffusionHeadAdaLayerNorm(config) for _ in range(config.num_head_layers))
        self.final_layer = VibeVoiceDiffusionHeadFinalLayer(
            config,
            output_size=config.audio_config.hidden_size,
        )

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.noisy_images_proj(noisy_latents)
        timestep_embedding = self.timestep_embedding(timesteps).to(condition.dtype)
        timestep_features = self.timestep_proj(timestep_embedding)
        condition = self.cond_proj(condition) + timestep_features
        for layer in self.layers:
            hidden_states = layer(hidden_states, condition)
        return self.final_layer(hidden_states, condition)


# Module-level so every VibeVoiceDiffusionSampler instance (one per model
# instance per rank) shares the resettable scheduler pool. Keyed by the full
# schedule contract to stay correct if multiple configs coexist in tests.
_SCHEDULER_STATE_CACHE: dict[tuple[int, int, str, str], tuple[Any, Any, Any]] = {}


@dataclass(frozen=True, slots=True)
class VibeVoiceDiffusionSampler:
    """Immutable configuration for one VibeVoice latent denoising loop."""

    num_train_timesteps: int
    beta_schedule: str
    prediction_type: str
    condition_size: int
    latent_size: int
    default_num_inference_steps: int

    def __post_init__(self) -> None:
        if self.num_train_timesteps < 1:
            raise ValueError("VibeVoice num_train_timesteps must be positive.")
        if self.condition_size < 1:
            raise ValueError("VibeVoice diffusion condition_size must be positive.")
        if self.latent_size < 1:
            raise ValueError("VibeVoice diffusion latent_size must be positive.")
        if self.default_num_inference_steps < 1:
            raise ValueError("VibeVoice default_num_inference_steps must be positive.")
        if self.default_num_inference_steps > self.num_train_timesteps:
            raise ValueError("VibeVoice default_num_inference_steps cannot exceed num_train_timesteps.")

    @classmethod
    def from_model_config(cls, config: Any) -> VibeVoiceDiffusionSampler:
        """Build the immutable sampling view from normalized VibeVoice config."""
        beta_schedule = str(getattr(config, "ddpm_beta_schedule", "squaredcos_cap_v2"))
        # Microsoft's scheduler accepts both spellings. Diffusers, used by the
        # Transformers PR runtime, exposes the canonical spelling only.
        if beta_schedule == "cosine":
            beta_schedule = "squaredcos_cap_v2"
        return cls(
            num_train_timesteps=int(getattr(config, "ddpm_num_steps", 1000)),
            beta_schedule=beta_schedule,
            prediction_type=str(getattr(config, "prediction_type", "v_prediction")),
            condition_size=int(config.hidden_size),
            latent_size=int(config.audio_config.hidden_size),
            default_num_inference_steps=int(getattr(config, "ddpm_num_inference_steps", 10)),
        )

    def create_scheduler(self) -> Any:
        """Return a fresh official-equivalent mutable DPM solver instance."""
        # Keep the model package import lightweight. Diffusers is an Omni
        # dependency, but only Stateful VibeVoice inference needs it here.
        from diffusers import DPMSolverMultistepScheduler

        return DPMSolverMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule=self.beta_schedule,
            prediction_type=self.prediction_type,
        )

    def acquire_scheduler(self, num_inference_steps: int) -> Any:
        """Return a scheduler reset to the exact post-``set_timesteps`` state.

        ``create_scheduler`` + ``set_timesteps`` costs ~0.5 ms per audio token
        in numpy schedule recomputation. The computed schedule is a
        deterministic function of the immutable sampler config and the step
        count, and this diffusers version's ``set_timesteps`` fully resets
        every mutable field (verified: ``sigmas``, ``timesteps``,
        ``num_inference_steps``, ``model_outputs``, ``lower_order_nums``,
        ``_step_index``, ``_begin_index``; ``flow_shift`` is only written
        when ``mu`` is passed, which this model never does). Restoring that
        exact state is bitwise identical to a fresh construction; the parity
        unit test pins this contract against the fresh-scheduler path.
        """
        key = (
            int(num_inference_steps),
            self.num_train_timesteps,
            self.beta_schedule,
            self.prediction_type,
        )
        entry = _SCHEDULER_STATE_CACHE.get(key)
        if entry is None:
            scheduler = self.create_scheduler()
            scheduler.set_timesteps(num_inference_steps=num_inference_steps)
            _SCHEDULER_STATE_CACHE[key] = (
                scheduler,
                scheduler.sigmas,
                scheduler.timesteps,
            )
            return scheduler
        scheduler, sigmas, timesteps = entry
        scheduler.sigmas = sigmas
        scheduler.timesteps = timesteps
        scheduler.num_inference_steps = len(timesteps)
        scheduler.model_outputs = [None] * scheduler.config.solver_order
        scheduler.lower_order_nums = 0
        scheduler._step_index = None
        scheduler._begin_index = None
        return scheduler

    def _resolve_num_inference_steps(
        self,
        num_inference_steps: int | None,
    ) -> int:
        steps = self.default_num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        if steps < 1:
            raise ValueError("VibeVoice num_inference_steps must be positive.")
        if steps > self.num_train_timesteps:
            raise ValueError(
                f"VibeVoice num_inference_steps cannot exceed num_train_timesteps={self.num_train_timesteps}."
            )
        return steps

    def _validate_inputs(
        self,
        positive_condition: torch.Tensor,
        negative_condition: torch.Tensor,
        noise: torch.Tensor,
        guidance_scale: float,
    ) -> int:
        if positive_condition.ndim != 2:
            raise ValueError(
                "VibeVoice positive_condition must have shape "
                f"(batch, hidden_size), got {tuple(positive_condition.shape)}."
            )
        if negative_condition.shape != positive_condition.shape:
            raise ValueError(
                "VibeVoice positive/negative condition shapes must match, got "
                f"{tuple(positive_condition.shape)} and "
                f"{tuple(negative_condition.shape)}."
            )
        if positive_condition.shape[0] < 1:
            raise ValueError("VibeVoice diffusion condition batch cannot be empty.")
        if positive_condition.shape[1] != self.condition_size:
            raise ValueError(
                "VibeVoice diffusion condition hidden size must be "
                f"{self.condition_size}, got {positive_condition.shape[1]}."
            )
        if not positive_condition.is_floating_point() or not negative_condition.is_floating_point():
            raise TypeError("VibeVoice diffusion conditions must be floating-point tensors.")
        if positive_condition.device != negative_condition.device:
            raise ValueError("VibeVoice positive/negative conditions must use the same device.")
        if positive_condition.dtype != negative_condition.dtype:
            raise ValueError("VibeVoice positive/negative conditions must use the same dtype.")

        batch_size = positive_condition.shape[0]
        expected_noise_shape = (batch_size * 2, self.latent_size)
        if tuple(noise.shape) != expected_noise_shape:
            raise ValueError(
                "VibeVoice diffusion noise must preserve the official cond/uncond "
                f"shape {expected_noise_shape}, got {tuple(noise.shape)}."
            )
        if not noise.is_floating_point():
            raise TypeError("VibeVoice diffusion noise must be a floating-point tensor.")
        if not math.isfinite(float(guidance_scale)):
            raise ValueError("VibeVoice guidance_scale must be finite.")
        return batch_size

    @torch.inference_mode()
    def sample_audio_latent(
        self,
        diffusion_head: nn.Module,
        positive_condition: torch.Tensor,
        negative_condition: torch.Tensor,
        noise: torch.Tensor,
        *,
        guidance_scale: float,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Denoise one continuous acoustic latent for each active request.

        The ``noise`` shape is ``(2 * batch, latent_size)`` to preserve the
        official implementation's RNG-consumption and solver-state semantics.
        Only its first half is fed to both CFG branches at every model step;
        the returned latent has shape ``(batch, 1, latent_size)``.
        """
        batch_size = self._validate_inputs(
            positive_condition,
            negative_condition,
            noise,
            guidance_scale,
        )
        steps = self._resolve_num_inference_steps(num_inference_steps)

        head_parameter = next(diffusion_head.parameters(), None)
        if head_parameter is not None:
            condition = torch.cat([positive_condition, negative_condition], dim=0).to(
                device=head_parameter.device, dtype=head_parameter.dtype
            )
        else:
            condition = torch.cat([positive_condition, negative_condition], dim=0)
        noisy_audio_latent = noise.to(condition).clone()

        # Match Microsoft/Transformers: timesteps and solver history start
        # fresh for every audio token (acquire_scheduler restores the exact
        # post-set_timesteps state); timestep batches are moved to the model
        # device only for the Diffusion Head invocation.
        scheduler = self.acquire_scheduler(steps)
        for timestep in scheduler.timesteps:
            shared_latent = noisy_audio_latent[:batch_size]
            combined_latent = torch.cat([shared_latent, shared_latent], dim=0)
            timestep_batch = timestep.repeat(combined_latent.shape[0]).to(combined_latent)
            prediction = diffusion_head(
                combined_latent,
                timestep_batch,
                condition,
            )
            if prediction.shape != combined_latent.shape:
                raise ValueError(
                    "VibeVoice Diffusion Head output shape must match its latent "
                    f"input, got {tuple(prediction.shape)} and "
                    f"{tuple(combined_latent.shape)}."
                )

            conditional_prediction = prediction[:batch_size]
            unconditional_prediction = prediction[batch_size:]
            guided_prediction = unconditional_prediction + float(guidance_scale) * (
                conditional_prediction - unconditional_prediction
            )
            solver_prediction = torch.cat([guided_prediction, guided_prediction], dim=0)
            noisy_audio_latent = scheduler.step(
                solver_prediction,
                timestep,
                noisy_audio_latent,
            ).prev_sample

        return noisy_audio_latent[:batch_size].unsqueeze(1)


__all__ = [
    "VibeVoiceDiffusionHead",
    "VibeVoiceDiffusionSampler",
    "VibeVoiceRMSNorm",
]
