# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deployment-only runtime configuration for VibeVoice stateful inference."""

from __future__ import annotations

import dataclasses
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class VibeVoiceRuntimeConfig:
    negative_kv_cache_memory_bytes: int = 4 * 1024**3
    negative_kv_activation_margin_bytes: int = 512 * 1024**2
    # Phase C1: replay the fixed-step DPM denoising loop through manual CUDA
    # graphs (bitwise identical to eager). Disable to force the eager loop.
    diffusion_cuda_graph: bool = True
    # Phase C2: replay the M4a decode (acoustic decoder + semantic encoder +
    # projectors) through a per-request manual CUDA graph (bitwise identical
    # to eager). Shares a graph pool with the C1 diffusion graph executor to
    # satisfy PyTorch's CUDACachingAllocator co-residency requirement.
    decode_cuda_graph: bool = True

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> VibeVoiceRuntimeConfig:
        additional_config = getattr(vllm_config, "additional_config", None)
        raw = additional_config.get("vibevoice_runtime_config") if isinstance(additional_config, dict) else None
        if raw is None:
            return cls()
        if hasattr(raw, "to_dict"):
            raw = raw.to_dict()
        elif not isinstance(raw, dict) and hasattr(raw, "__dict__"):
            raw = vars(raw)
        if not isinstance(raw, dict):
            logger.warning(
                "Ignoring invalid vibevoice_runtime_config=%r; expected a dict.",
                raw,
            )
            return cls()

        field_types = {field.name: type(field.default) for field in dataclasses.fields(cls)}
        values: dict[str, Any] = {}
        for key, value in raw.items():
            if key not in field_types:
                logger.warning(
                    "Ignoring unknown VibeVoice runtime config key: %s",
                    key,
                )
                continue
            if field_types[key] is bool:
                if not isinstance(value, bool):
                    raise ValueError(f"VibeVoice runtime config {key} must be a bool, got {value!r}.")
                values[key] = value
                continue
            if isinstance(value, bool):
                raise ValueError(f"VibeVoice runtime config {key} must be an integer, not bool.")
            try:
                values[key] = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"VibeVoice runtime config {key} must be an integer, got {value!r}.") from exc

        result = cls(**values)
        if result.negative_kv_cache_memory_bytes <= 0:
            raise ValueError("negative_kv_cache_memory_bytes must be positive.")
        if result.negative_kv_activation_margin_bytes < 0:
            raise ValueError("negative_kv_activation_margin_bytes must be non-negative.")
        return result


__all__ = ["VibeVoiceRuntimeConfig"]
