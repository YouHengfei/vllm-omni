# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from Microsoft VibeVoice and Transformers PR #40546.
"""Model-local waveform decode and semantic feedback for one audio token.

The kernel is intentionally stateless: causal Acoustic Decoder and Semantic
Encoder caches are supplied by the caller and returned in the result. Request
ownership, dynamic-batch cache packing, cleanup, and waveform serving belong to
M4c rather than this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(slots=True)
class VibeVoiceAudioTokenDecodeOutput:
    """Outputs needed by the next AR step and future request state."""

    audio: torch.Tensor
    semantic_latent: torch.Tensor
    next_embedding: torch.Tensor
    acoustic_cache: Any
    semantic_cache: Any


@dataclass(frozen=True, slots=True)
class VibeVoiceAudioTokenDecoder:
    """Immutable shape/config view for VibeVoice audio-token decoding."""

    latent_size: int
    semantic_size: int
    condition_size: int
    audio_channels: int
    samples_per_token: int

    def __post_init__(self) -> None:
        for name in (
            "latent_size",
            "semantic_size",
            "condition_size",
            "audio_channels",
            "samples_per_token",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"VibeVoice {name} must be positive.")

    @classmethod
    def from_model_config(cls, config: Any) -> "VibeVoiceAudioTokenDecoder":
        decoder_config = config.audio_config.decoder_config
        upsampling_ratios = tuple(decoder_config.upsampling_ratios)
        if not upsampling_ratios:
            raise ValueError(
                "VibeVoice Acoustic Decoder upsampling_ratios cannot be empty."
            )
        return cls(
            latent_size=int(config.audio_config.hidden_size),
            semantic_size=int(config.semantic_model_config.hidden_size),
            condition_size=int(config.hidden_size),
            audio_channels=int(decoder_config.channels),
            samples_per_token=math.prod(int(ratio) for ratio in upsampling_ratios),
        )

    def _validate_latent(self, audio_latent: torch.Tensor) -> int:
        if audio_latent.ndim != 3:
            raise ValueError(
                "VibeVoice audio_latent must have shape "
                f"(batch, 1, {self.latent_size}), got "
                f"{tuple(audio_latent.shape)}."
            )
        if audio_latent.shape[0] < 1:
            raise ValueError("VibeVoice audio_latent batch cannot be empty.")
        if audio_latent.shape[1:] != (1, self.latent_size):
            raise ValueError(
                "VibeVoice audio_latent must have shape "
                f"(batch, 1, {self.latent_size}), got "
                f"{tuple(audio_latent.shape)}."
            )
        if not audio_latent.is_floating_point():
            raise TypeError("VibeVoice audio_latent must be a floating-point tensor.")
        return audio_latent.shape[0]

    @staticmethod
    def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
        parameter = next(module.parameters(), None)
        if parameter is None:
            raise ValueError(
                f"VibeVoice decode module {type(module).__name__} has no parameters."
            )
        return parameter.device, parameter.dtype

    @staticmethod
    def _validate_factor(name: str, factor: torch.Tensor) -> None:
        if factor.numel() != 1 or not factor.is_floating_point():
            raise ValueError(
                f"VibeVoice {name} must be one floating-point scalar tensor."
            )

    @torch.inference_mode()
    def decode_audio_token(
        self,
        *,
        audio_tower: nn.Module,
        semantic_encoder: nn.Module,
        acoustic_projector: nn.Module,
        semantic_connector: nn.Module,
        latent_scaling_factor: torch.Tensor,
        latent_bias_factor: torch.Tensor,
        audio_latent: torch.Tensor,
        acoustic_cache: Any = None,
        semantic_cache: Any = None,
    ) -> VibeVoiceAudioTokenDecodeOutput:
        """Decode one latent token and build the next Qwen input embedding."""
        batch_size = self._validate_latent(audio_latent)
        self._validate_factor("latent_scaling_factor", latent_scaling_factor)
        self._validate_factor("latent_bias_factor", latent_bias_factor)

        tower_device, tower_dtype = self._module_device_dtype(audio_tower)
        audio_latent = audio_latent.to(device=tower_device, dtype=tower_dtype)
        decoder_latent = (
            audio_latent / latent_scaling_factor.to(audio_latent)
            - latent_bias_factor.to(audio_latent)
        )
        decoder_output = audio_tower.decode(
            decoder_latent,
            padding_cache=acoustic_cache,
            use_cache=True,
        )
        audio = getattr(decoder_output, "audio", None)
        next_acoustic_cache = getattr(decoder_output, "padding_cache", None)
        expected_audio_shape = (
            batch_size,
            self.audio_channels,
            self.samples_per_token,
        )
        if not isinstance(audio, torch.Tensor) or tuple(audio.shape) != expected_audio_shape:
            actual_shape = tuple(audio.shape) if isinstance(audio, torch.Tensor) else None
            raise ValueError(
                "VibeVoice Acoustic Decoder output must have shape "
                f"{expected_audio_shape}, got {actual_shape}."
            )
        if next_acoustic_cache is None:
            raise ValueError(
                "VibeVoice Acoustic Decoder did not return a causal padding cache."
            )

        semantic_device, semantic_dtype = self._module_device_dtype(semantic_encoder)
        semantic_output = semantic_encoder(
            audio.to(device=semantic_device, dtype=semantic_dtype),
            padding_cache=semantic_cache,
            use_cache=True,
        )
        semantic_latent = getattr(semantic_output, "latents", None)
        next_semantic_cache = getattr(semantic_output, "padding_cache", None)
        expected_semantic_shape = (batch_size, 1, self.semantic_size)
        if (
            not isinstance(semantic_latent, torch.Tensor)
            or tuple(semantic_latent.shape) != expected_semantic_shape
        ):
            actual_shape = (
                tuple(semantic_latent.shape)
                if isinstance(semantic_latent, torch.Tensor)
                else None
            )
            raise ValueError(
                "VibeVoice Semantic Encoder output must have shape "
                f"{expected_semantic_shape}, got {actual_shape}."
            )
        if next_semantic_cache is None:
            raise ValueError(
                "VibeVoice Semantic Encoder did not return a causal padding cache."
            )

        acoustic_device, acoustic_dtype = self._module_device_dtype(
            acoustic_projector
        )
        acoustic_embedding = acoustic_projector(
            audio_latent.to(device=acoustic_device, dtype=acoustic_dtype)
        )
        semantic_connector_device, semantic_connector_dtype = (
            self._module_device_dtype(semantic_connector)
        )
        semantic_embedding = semantic_connector(
            semantic_latent.to(
                device=semantic_connector_device,
                dtype=semantic_connector_dtype,
            )
        ).to(acoustic_embedding)
        expected_embedding_shape = (batch_size, 1, self.condition_size)
        if tuple(acoustic_embedding.shape) != expected_embedding_shape:
            raise ValueError(
                "VibeVoice acoustic projector output must have shape "
                f"{expected_embedding_shape}, got "
                f"{tuple(acoustic_embedding.shape)}."
            )
        if tuple(semantic_embedding.shape) != expected_embedding_shape:
            raise ValueError(
                "VibeVoice semantic connector output must have shape "
                f"{expected_embedding_shape}, got "
                f"{tuple(semantic_embedding.shape)}."
            )

        return VibeVoiceAudioTokenDecodeOutput(
            audio=audio,
            semantic_latent=semantic_latent,
            next_embedding=acoustic_embedding + semantic_embedding,
            acoustic_cache=next_acoustic_cache,
            semantic_cache=next_semantic_cache,
        )


__all__ = [
    "VibeVoiceAudioTokenDecodeOutput",
    "VibeVoiceAudioTokenDecoder",
]
