# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request state for VibeVoice's single-stage AR decode loop.

This module deliberately owns no scheduler or PagedAttention storage. The
positive branch remains runner-owned vLLM Qwen2 state. A future negative-branch
owner must publish the corresponding hidden condition through
:meth:`record_negative_condition`; until then an audio-token transition fails
explicitly instead of silently running unguided diffusion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from .audio_decode import VibeVoiceAudioTokenDecodeOutput


class VibeVoiceInferenceKernel(Protocol):
    """Model-side math used by :class:`VibeVoiceStatefulInference`."""

    def sample_audio_latent(
        self,
        positive_condition: torch.Tensor,
        negative_condition: torch.Tensor,
        noise: torch.Tensor,
        *,
        guidance_scale: float,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor: ...

    def decode_audio_token(
        self,
        audio_latent: torch.Tensor,
        *,
        acoustic_cache: Any = None,
        semantic_cache: Any = None,
    ) -> VibeVoiceAudioTokenDecodeOutput: ...


class VibeVoiceNegativeKVBranch(Protocol):
    """Required ownership boundary for the future negative Qwen branch.

    The implementation must own independent PagedAttention KV, advance only on
    VibeVoice audio-generation inputs, reset to a one-token audio-BOS context at
    every audio segment, and clean up with the parent request. It must not keep
    cache tensors in :class:`VibeVoiceStatefulInference`.
    """

    def reset_audio_segment(self, request_id: str) -> None: ...

    def forward_step(
        self,
        request_ids: list[str],
        input_embeddings: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Advance each negative branch by one embedding and return hidden rows."""
        ...

    def on_requests_finished(self, request_ids: set[str] | list[str]) -> None: ...


@dataclass(slots=True)
class VibeVoiceRequestState:
    """Model-local non-Qwen state for one parent request."""

    request_id: str
    guidance_scale: float
    num_diffusion_steps: int
    acoustic_cache: Any = None
    semantic_cache: Any = None
    positive_condition: torch.Tensor | None = None
    negative_condition: torch.Tensor | None = None
    negative_input_embedding: torch.Tensor | None = None
    next_embedding: torch.Tensor | None = None
    waveform_chunks_cpu: list[torch.Tensor] = field(default_factory=list)
    in_audio_segment: bool = False
    negative_reset_pending: bool = False
    audio_token_count: int = 0

    def clear(self) -> None:
        self.acoustic_cache = None
        self.semantic_cache = None
        self.positive_condition = None
        self.negative_condition = None
        self.negative_input_embedding = None
        self.next_embedding = None
        self.waveform_chunks_cpu.clear()
        self.in_audio_segment = False
        self.negative_reset_pending = False
        self.audio_token_count = 0


class da:
    """Request-indexed state machine around the frozen M4a/M4b kernels.

    Convolution caches and waveform chunks are parent-request state. Qwen KV is
    intentionally absent: positive KV belongs to ``GPUARModelRunner`` and the
    unresolved negative PagedAttention branch must have a separate owner.
    """

    def __init__(
        self,
        *,
        audio_bos_token_id: int,
        audio_eos_token_id: int,
        audio_token_id: int,
        eos_token_id: int,
        latent_size: int,
        condition_size: int,
        default_guidance_scale: float,
        default_num_diffusion_steps: int,
    ) -> None:
        token_ids = {
            "audio_bos_token_id": audio_bos_token_id,
            "audio_eos_token_id": audio_eos_token_id,
            "audio_token_id": audio_token_id,
            "eos_token_id": eos_token_id,
        }
        if len(set(token_ids.values())) != len(token_ids):
            raise ValueError(f"VibeVoice control token IDs must be distinct, got {token_ids}.")
        if latent_size < 1 or condition_size < 1:
            raise ValueError("VibeVoice latent_size and condition_size must be positive.")
        self.audio_bos_token_id = int(audio_bos_token_id)
        self.audio_eos_token_id = int(audio_eos_token_id)
        self.audio_token_id = int(audio_token_id)
        self.eos_token_id = int(eos_token_id)
        self.latent_size = int(latent_size)
        self.condition_size = int(condition_size)
        self.default_guidance_scale = self._validate_guidance_scale(
            default_guidance_scale
        )
        self.default_num_diffusion_steps = self._validate_num_diffusion_steps(
            default_num_diffusion_steps
        )
        self._states: dict[str, VibeVoiceRequestState] = {}
        self._deferred_cleanup_ids: set[str] = set()

    @staticmethod
    def _validate_guidance_scale(value: Any) -> float:
        try:
            scale = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"VibeVoice guidance_scale must be finite, got {value!r}."
            ) from exc
        if not math.isfinite(scale):
            raise ValueError(
                f"VibeVoice guidance_scale must be finite, got {value!r}."
            )
        return scale

    @staticmethod
    def _validate_num_diffusion_steps(value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError(
                "VibeVoice num_diffusion_steps must be a positive integer."
            )
        try:
            steps = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "VibeVoice num_diffusion_steps must be a positive integer."
            ) from exc
        if steps < 1:
            raise ValueError(
                "VibeVoice num_diffusion_steps must be a positive integer."
            )
        return steps

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return tuple(self._states)

    @property
    def deferred_cleanup_ids(self) -> frozenset[str]:
        return frozenset(self._deferred_cleanup_ids)

    def get_or_create(
        self,
        request_id: str,
        *,
        reset: bool = False,
    ) -> VibeVoiceRequestState:
        if not request_id:
            raise ValueError("VibeVoice request_id must be non-empty.")
        if reset:
            self.cleanup_request(request_id)
        state = self._states.get(request_id)
        if state is None:
            state = VibeVoiceRequestState(
                request_id=request_id,
                guidance_scale=self.default_guidance_scale,
                num_diffusion_steps=self.default_num_diffusion_steps,
            )
            self._states[request_id] = state
        return state

    def get(self, request_id: str) -> VibeVoiceRequestState | None:
        return self._states.get(request_id)

    def set_runtime_controls(
        self,
        request_id: str,
        extra_args: dict[str, Any] | None,
    ) -> None:
        if not extra_args:
            return
        state = self.get_or_create(request_id)
        if "guidance_scale" in extra_args:
            state.guidance_scale = self._validate_guidance_scale(
                extra_args["guidance_scale"]
            )
        if "num_diffusion_steps" in extra_args:
            state.num_diffusion_steps = self._validate_num_diffusion_steps(
                extra_args["num_diffusion_steps"]
            )

    def _validate_condition(
        self,
        name: str,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(condition, torch.Tensor):
            raise TypeError(f"VibeVoice {name} must be a tensor.")
        expected_shape = (1, self.condition_size)
        if tuple(condition.shape) != expected_shape:
            raise ValueError(
                f"VibeVoice {name} must have shape {expected_shape}, got "
                f"{tuple(condition.shape)}."
            )
        if not condition.is_floating_point():
            raise TypeError(f"VibeVoice {name} must be floating-point.")
        return condition.detach().contiguous()

    def record_positive_condition(
        self,
        request_id: str,
        condition: torch.Tensor,
    ) -> None:
        state = self.get_or_create(request_id)
        state.positive_condition = self._validate_condition(
            "positive_condition", condition
        )

    def record_negative_input_embedding(
        self,
        request_id: str,
        input_embedding: torch.Tensor,
    ) -> None:
        state = self.get_or_create(request_id)
        state.negative_input_embedding = self._validate_condition(
            "negative_input_embedding", input_embedding
        )

    def record_negative_condition(
        self,
        request_id: str,
        condition: torch.Tensor,
    ) -> None:
        state = self.get_or_create(request_id)
        state.negative_condition = self._validate_condition(
            "negative_condition", condition
        )
        state.negative_reset_pending = False

    def start_audio_segment(
        self,
        request_id: str,
        negative_kv_branch: VibeVoiceNegativeKVBranch | None = None,
    ) -> None:
        state = self.get_or_create(request_id)
        self._start_audio_segment(state, negative_kv_branch)

    def _start_audio_segment(
        self,
        state: VibeVoiceRequestState,
        negative_kv_branch: VibeVoiceNegativeKVBranch | None,
    ) -> None:
        state.in_audio_segment = True
        state.positive_condition = None
        state.negative_condition = None
        state.negative_input_embedding = None
        state.negative_reset_pending = True
        if negative_kv_branch is not None:
            negative_kv_branch.reset_audio_segment(state.request_id)

    def _finish_audio_segment(self, state: VibeVoiceRequestState) -> None:
        state.in_audio_segment = False
        state.positive_condition = None
        state.negative_condition = None
        state.negative_input_embedding = None
        state.negative_reset_pending = False

    def process_sampled_token(
        self,
        *,
        request_id: str,
        token_id: int,
        token_embedding: torch.Tensor,
        kernel: VibeVoiceInferenceKernel,
        negative_kv_branch: VibeVoiceNegativeKVBranch | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply one sampled control-token transition before the next Qwen step."""
        state = self.get_or_create(request_id)
        if token_embedding.ndim != 2 or token_embedding.shape[0] != 1:
            raise ValueError(
                "VibeVoice token_embedding must have shape (1, hidden_size), "
                f"got {tuple(token_embedding.shape)}."
            )
        if token_embedding.shape[1] != self.condition_size:
            raise ValueError(
                "VibeVoice token_embedding hidden size must be "
                f"{self.condition_size}, got {token_embedding.shape[1]}."
            )

        token_id = int(token_id)
        if token_id == self.audio_bos_token_id:
            self._start_audio_segment(state, negative_kv_branch)
            state.next_embedding = token_embedding
            return token_embedding, None
        if token_id == self.audio_eos_token_id:
            self._finish_audio_segment(state)
            state.next_embedding = token_embedding
            return token_embedding, None
        if token_id == self.eos_token_id:
            self._finish_audio_segment(state)
            state.next_embedding = token_embedding
            return token_embedding, None
        if token_id != self.audio_token_id:
            raise ValueError(
                f"Unsupported VibeVoice control token ID {token_id}; expected one of "
                f"{self.audio_bos_token_id}, {self.audio_eos_token_id}, "
                f"{self.audio_token_id}, {self.eos_token_id}."
            )
        next_embeddings, audio_chunks = self.process_audio_tokens_batch(
            request_ids=[request_id],
            token_embeddings=[token_embedding],
            kernel=kernel,
        )
        return next_embeddings[0], audio_chunks[0]

    def process_audio_tokens_batch(
        self,
        *,
        request_ids: list[str],
        token_embeddings: list[torch.Tensor],
        kernel: VibeVoiceInferenceKernel,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Batch M4a over the active subset, then decode per-request caches."""
        if not request_ids:
            return [], []
        if len(request_ids) != len(token_embeddings):
            raise ValueError(
                "VibeVoice audio-token request/embedding batch lengths must match."
            )
        if len(request_ids) != len(set(request_ids)):
            raise ValueError(
                "VibeVoice audio-token active subset contains duplicate request IDs."
            )

        states: list[VibeVoiceRequestState] = []
        positive_conditions: list[torch.Tensor] = []
        negative_conditions: list[torch.Tensor] = []
        guidance_scale: float | None = None
        num_diffusion_steps: int | None = None
        for request_id, token_embedding in zip(
            request_ids,
            token_embeddings,
            strict=True,
        ):
            state = self.get_or_create(request_id)
            if token_embedding.shape != (1, self.condition_size):
                raise ValueError(
                    "VibeVoice token_embedding must have shape "
                    f"(1, {self.condition_size}), got "
                    f"{tuple(token_embedding.shape)}."
                )
            if not state.in_audio_segment:
                raise RuntimeError(
                    "VibeVoice audio_token was received outside an audio segment; "
                    "audio_bos_token must be generated first."
                )
            if state.positive_condition is None:
                raise RuntimeError(
                    "VibeVoice audio_token has no positive Qwen condition from the "
                    "preceding AR step."
                )
            if state.negative_condition is None or state.negative_reset_pending:
                raise RuntimeError(
                    "VibeVoice audio_token requires an independent negative Qwen "
                    "PagedAttention branch. No aligned negative condition is bound; "
                    "unguided fallback is intentionally disabled."
                )
            if guidance_scale is None:
                guidance_scale = state.guidance_scale
                num_diffusion_steps = state.num_diffusion_steps
            elif (
                state.guidance_scale != guidance_scale
                or state.num_diffusion_steps != num_diffusion_steps
            ):
                raise RuntimeError(
                    "VibeVoice active audio-token requests with different guidance_scale "
                    "or num_diffusion_steps cannot share one diffusion batch."
                )
            positive = state.positive_condition
            negative = state.negative_condition.to(positive)
            if positive_conditions:
                if positive.device != positive_conditions[0].device:
                    raise ValueError(
                        "VibeVoice active diffusion conditions must use one device."
                    )
                if positive.dtype != positive_conditions[0].dtype:
                    raise ValueError(
                        "VibeVoice active diffusion conditions must use one dtype."
                    )
            states.append(state)
            positive_conditions.append(positive)
            negative_conditions.append(negative)

        positive_batch = torch.cat(positive_conditions, dim=0)
        negative_batch = torch.cat(negative_conditions, dim=0)
        batch_size = len(states)
        # Preserve official active-subset RNG ordering: one [2B, latent] draw,
        # not B independent [2, latent] draws interleaved per request.
        noise = torch.randn(
            (2 * batch_size, self.latent_size),
            device=positive_batch.device,
            dtype=positive_batch.dtype,
        )
        audio_latents = kernel.sample_audio_latent(
            positive_batch,
            negative_batch,
            noise,
            guidance_scale=guidance_scale,
            num_inference_steps=num_diffusion_steps,
        )
        expected_latent_shape = (batch_size, 1, self.latent_size)
        if tuple(audio_latents.shape) != expected_latent_shape:
            raise ValueError(
                "VibeVoice stateful diffusion output must have shape "
                f"{expected_latent_shape}, got {tuple(audio_latents.shape)}."
            )

        next_embeddings: list[torch.Tensor] = []
        audio_chunks: list[torch.Tensor] = []
        for index, state in enumerate(states):
            decoded = kernel.decode_audio_token(
                audio_latents[index : index + 1],
                acoustic_cache=state.acoustic_cache,
                semantic_cache=state.semantic_cache,
            )
            state.acoustic_cache = decoded.acoustic_cache
            state.semantic_cache = decoded.semantic_cache
            state.next_embedding = decoded.next_embedding.reshape(1, -1)
            state.waveform_chunks_cpu.append(
                decoded.audio.detach()
                .reshape(-1)
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
            )
            state.audio_token_count += 1
            # Conditions are one-step values. Keeping either one would allow a
            # desynchronized branch to be reused silently on the next token.
            state.positive_condition = None
            state.negative_condition = None
            next_embeddings.append(state.next_embedding)
            audio_chunks.append(decoded.audio)
        return next_embeddings, audio_chunks

    def on_requests_finished(
        self,
        request_ids: set[str] | list[str],
    ) -> None:
        # GPUARModelRunner calls this before the current forward. Delay cleanup
        # so a request that is both finished and scheduled can still complete
        # its final model step.
        self._deferred_cleanup_ids.update(request_ids)

    def flush_deferred_cleanup(
        self,
        *,
        exclude_request_ids: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        cleanup_ids = self._deferred_cleanup_ids - set(exclude_request_ids)
        for request_id in cleanup_ids:
            self.cleanup_request(request_id)
        self._deferred_cleanup_ids.difference_update(cleanup_ids)

    def finish_postprocess(self, request_id: str) -> None:
        if request_id in self._deferred_cleanup_ids:
            self.cleanup_request(request_id)

    def cleanup_request(self, request_id: str) -> None:
        state = self._states.pop(request_id, None)
        if state is not None:
            state.clear()
        self._deferred_cleanup_ids.discard(request_id)

    def clear(self) -> None:
        for request_id in tuple(self._states):
            self.cleanup_request(request_id)
        self._deferred_cleanup_ids.clear()


__all__ = [
    "VibeVoiceInferenceKernel",
    "VibeVoiceNegativeKVBranch",
    "VibeVoiceRequestState",
    "VibeVoiceStatefulInference",
]
