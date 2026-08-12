# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VibeVoice executor for the runner-owned negative causal KV branch."""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.sequence import IntermediateTensors

from vllm_omni.worker.named_kv_branch import NamedCausalKVBranch


class VibeVoiceNegativeBranch:
    """Advance official negative-Qwen state without owning its Paged KV."""

    def __init__(
        self,
        *,
        store: NamedCausalKVBranch,
        language_model: Qwen2Model,
        hidden_size: int,
    ) -> None:
        if store.name != "negative":
            raise ValueError(
                "VibeVoice requires a named KV branch called 'negative', got "
                f"{store.name!r}."
            )
        if hidden_size < 1:
            raise ValueError("VibeVoice negative hidden_size must be positive.")
        self.store = store
        self.language_model = language_model
        self.hidden_size = int(hidden_size)

    def reset_audio_segment(self, request_id: str) -> None:
        self.store.reset(request_id)

    def forward_step(
        self,
        request_ids: list[str],
        input_embeddings: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if len(request_ids) != 1 or len(input_embeddings) != 1:
            raise ValueError(
                "VibeVoice negative KV v1 requires exactly one active request."
            )
        request_id = request_ids[0]
        embedding = input_embeddings[0]
        if not isinstance(embedding, torch.Tensor):
            raise TypeError("VibeVoice negative input embedding must be a tensor.")
        if tuple(embedding.shape) != (1, self.hidden_size):
            raise ValueError(
                "VibeVoice negative input embedding must have shape "
                f"(1, {self.hidden_size}), got {tuple(embedding.shape)}."
            )
        if not embedding.is_floating_point():
            raise TypeError(
                "VibeVoice negative input embedding must be floating-point."
            )

        try:
            with self.store.append_and_enter(request_id) as step:
                hidden_states: Any = self.language_model(
                    input_ids=None,
                    positions=step.position,
                    inputs_embeds=embedding,
                )
                if isinstance(hidden_states, IntermediateTensors):
                    raise RuntimeError(
                        "VibeVoice negative Qwen returned pipeline intermediate tensors; "
                        "PP=1 is required."
                    )
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                if not isinstance(hidden_states, torch.Tensor):
                    raise TypeError(
                        "VibeVoice negative Qwen must return hidden-state tensor output."
                    )
                if tuple(hidden_states.shape) != (1, self.hidden_size):
                    raise ValueError(
                        "VibeVoice negative Qwen hidden state must have shape "
                        f"(1, {self.hidden_size}), got {tuple(hidden_states.shape)}."
                    )
                condition = hidden_states.detach().contiguous()
        except Exception:
            # NamedCausalKVBranch already drops a branch after failures inside
            # its context. Keep this idempotent free so validation/context
            # setup failures have the same request-level cleanup contract.
            self.free(request_id)
            raise
        return [condition]

    def free(self, request_id: str) -> None:
        self.store.free(request_id)


__all__ = ["VibeVoiceNegativeBranch"]
