# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VibeVoice executor for the model-owned negative causal KV branch.

.. note:: Downgrade adaptation (Plan B)

    Upstream (``fix/vibevoice-review-remediation`` and later) advances the
    negative Qwen branch through a *runner-owned* ``NamedCausalKVBranch`` that
    shares the backbone's ``Attention`` layers and swaps their paged KV via a
    forward-context override. That mechanism depends on runner capabilities
    (``vllm_omni.worker.named_kv_branch``) introduced by the
    "request-owned AR runner capabilities" framework change.

    This branch runs on the older ``support-tecoomni`` baseline, which lacks
    that runner feature. To keep *zero* framework changes, the negative KV is
    instead owned entirely by the model: per-request, per-layer K/V buffers
    live here, and each negative step is a manual eager forward that reuses the
    *pure* Qwen2 submodules (``qkv_proj`` / ``rotary_emb`` / ``o_proj`` /
    ``*_layernorm`` / ``mlp`` / final ``norm``) while bypassing the vLLM
    ``Attention`` op (the only forward-context/KV-coupled component).

    The public surface intentionally mirrors the upstream
    ``VibeVoiceNegativeKVBranch`` protocol (``reset_audio_segment`` /
    ``forward_step`` / ``free``) so that migrating back to the runner-owned
    implementation is a mechanical swap: delete this store and re-bind the
    upstream ``NamedCausalKVBranch`` in the model. See ``stateful.py`` for the
    protocol contract.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.qwen2 import Qwen2Model

logger = init_logger(__name__)


class VibeVoiceSelfManagedNegativeKVStore:
    """Model-owned negative Qwen KV store (no runner / paged-attention coupling).

    Implements the ``VibeVoiceNegativeKVBranch`` protocol by owning one growing
    K/V buffer per (request, decoder layer) and advancing the shared Qwen2
    backbone with a manual eager attention. Weight sharing with the positive
    branch is preserved because only the projection / normalization submodules
    are reused; the vLLM ``Attention`` op (which owns the positive paged KV via
    the runner forward context) is never invoked on this path.
    """

    def __init__(
        self,
        *,
        language_model: Qwen2Model,
        hidden_size: int,
    ) -> None:
        if hidden_size < 1:
            raise ValueError("VibeVoice negative hidden_size must be positive.")
        self.language_model = language_model
        self.hidden_size = int(hidden_size)
        # request_id -> per-layer K/V buffers, each (seq_len, num_kv_heads, head_dim).
        self._k_buffers: dict[str, list[torch.Tensor]] = {}
        self._v_buffers: dict[str, list[torch.Tensor]] = {}
        # request_id -> current negative sequence length.
        self._seq_lens: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Protocol surface (mirrors upstream VibeVoiceNegativeKVBranch)
    # ------------------------------------------------------------------
    def reset_audio_segment(self, request_id: str) -> None:
        """Reset the negative branch to an empty (pre-BOS) context.

        The next :meth:`forward_step` for ``request_id`` lands at position 0,
        establishing the one-token audio-BOS context, matching the upstream
        ``NamedCausalKVBranch.reset`` semantics.
        """
        self._drop(request_id)
        self._seq_lens[request_id] = 0

    def free(self, request_id: str) -> None:
        self._drop(request_id)

    def forward_step(
        self,
        request_ids: list[str],
        input_embeddings: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Advance each negative branch by one embedding; return hidden rows."""
        if not request_ids or len(request_ids) != len(input_embeddings):
            raise ValueError("VibeVoice negative KV request and embedding batches must be non-empty and aligned.")
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("VibeVoice negative KV batch contains duplicate request IDs.")

        for request_id, embedding in zip(request_ids, input_embeddings, strict=True):
            if request_id not in self._seq_lens:
                raise RuntimeError(
                    f"VibeVoice negative branch request {request_id!r} must be reset before forward_step."
                )
            if not isinstance(embedding, torch.Tensor):
                raise TypeError("VibeVoice negative input embedding must be a tensor.")
            if tuple(embedding.shape) != (1, self.hidden_size):
                raise ValueError(
                    "VibeVoice negative input embedding must have shape "
                    f"(1, {self.hidden_size}), got {tuple(embedding.shape)}."
                )
            if not embedding.is_floating_point():
                raise TypeError("VibeVoice negative input embedding must be floating-point.")

        try:
            hidden = self._negative_forward(request_ids, input_embeddings)
            # Only advance positions after the whole logical batch succeeded so
            # a mid-batch fault never leaves some requests advanced and others
            # not (matching the upstream append-then-enter contract).
            for request_id in request_ids:
                self._seq_lens[request_id] += 1
        except Exception:
            # A model-forward exception is fatal to the current engine. Drop
            # every request touched by this logical batch so no partially
            # advanced negative branch survives into shutdown diagnostics.
            for request_id in request_ids:
                self.free(request_id)
            raise

        if tuple(hidden.shape) != (len(request_ids), self.hidden_size):
            raise ValueError(
                "VibeVoice negative Qwen hidden state must have shape "
                f"({len(request_ids)}, {self.hidden_size}), got {tuple(hidden.shape)}."
            )
        # Own the batch until the caller binds every row to request-local
        # state (which clones again per request).
        owned = hidden.detach().clone(memory_format=torch.contiguous_format)
        return [row.reshape(1, self.hidden_size) for row in owned.unbind(0)]

    # ------------------------------------------------------------------
    # Internal manual forward
    # ------------------------------------------------------------------
    def _drop(self, request_id: str) -> None:
        self._k_buffers.pop(request_id, None)
        self._v_buffers.pop(request_id, None)
        self._seq_lens.pop(request_id, None)

    def _ensure_buffers(self, request_id: str, *, device: torch.device, dtype: torch.dtype) -> None:
        if request_id in self._k_buffers:
            return
        layers = self.language_model.layers
        num_layers = len(layers)
        self._k_buffers[request_id] = [torch.empty(0, device=device, dtype=dtype) for _ in range(num_layers)]
        self._v_buffers[request_id] = [torch.empty(0, device=device, dtype=dtype) for _ in range(num_layers)]

    def _negative_forward(
        self,
        request_ids: list[str],
        input_embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        """Run the shared Qwen2 stack; attention reads model-owned KV buffers."""
        first = input_embeddings[0]
        device, dtype = first.device, first.dtype
        for request_id in request_ids:
            self._ensure_buffers(request_id, device=device, dtype=dtype)

        # Current token position per request (index of the token being added).
        positions = torch.tensor(
            [self._seq_lens[request_id] for request_id in request_ids],
            dtype=torch.long,
            device=device,
        )

        hidden = torch.cat(input_embeddings, dim=0).to(device=device, dtype=dtype)
        language_model = self.language_model
        start_layer = int(getattr(language_model, "start_layer", 0))
        end_layer = int(getattr(language_model, "end_layer", len(language_model.layers)))

        residual: torch.Tensor | None = None
        for layer_index, layer in enumerate(language_model.layers):
            if not (start_layer <= layer_index < end_layer):
                continue
            hidden, residual = self._decoder_layer_forward(
                layer,
                layer_index,
                positions,
                hidden,
                residual,
                request_ids,
            )

        hidden, _ = language_model.norm(hidden, residual)
        return hidden

    def _decoder_layer_forward(
        self,
        layer: Any,
        layer_index: int,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        request_ids: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replicate ``Qwen2DecoderLayer.forward`` with a manual negative attention."""
        if residual is None:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
        else:
            hidden, residual = layer.input_layernorm(hidden, residual)
        hidden = self._manual_attention(layer.self_attn, layer_index, positions, hidden, request_ids)
        hidden, residual = layer.post_attention_layernorm(hidden, residual)
        hidden = layer.mlp(hidden)
        return hidden, residual

    def _manual_attention(
        self,
        attn: Any,
        layer_index: int,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        request_ids: list[str],
    ) -> torch.Tensor:
        """Replicate ``Qwen2Attention.forward`` with model-owned KV + eager SDPA.

        Only ``self.attn(q, k, v)`` (the vLLM paged-attention op) is replaced;
        the qkv projection, optional QK-norm, rotary embedding, and output
        projection are reused bit-for-bit from the shared layer.
        """
        qkv, _ = attn.qkv_proj(hidden)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)

        num_heads = int(attn.num_heads)
        num_kv_heads = int(attn.num_kv_heads)
        head_dim = int(attn.head_dim)

        # Optional QK normalization (standard Qwen2 leaves this disabled).
        if getattr(attn, "qk_norm", False):
            batch = q.shape[0]
            q = attn.q_norm(q.view(batch, num_heads, head_dim))
            k = attn.k_norm(k.view(batch, num_kv_heads, head_dim))

        q, k = attn.rotary_emb(positions, q, k)

        q = q.view(-1, num_heads, head_dim)
        k = k.view(-1, num_kv_heads, head_dim)
        v = v.view(-1, num_kv_heads, head_dim)

        scaling = float(attn.scaling)
        out = torch.empty_like(q)
        for row, request_id in enumerate(request_ids):
            # Append the new token's K/V, then attend to the whole history
            # (previous tokens plus the current one).
            new_k = k[row].unsqueeze(0)  # (1, num_kv_heads, head_dim)
            new_v = v[row].unsqueeze(0)
            k_hist = torch.cat([self._k_buffers[request_id][layer_index], new_k], dim=0)
            v_hist = torch.cat([self._v_buffers[request_id][layer_index], new_v], dim=0)
            self._k_buffers[request_id][layer_index] = k_hist
            self._v_buffers[request_id][layer_index] = v_hist

            # (L, num_kv_heads, head_dim) -> (num_kv_heads, L, head_dim)
            key = k_hist.permute(1, 0, 2)
            value = v_hist.permute(1, 0, 2)
            # GQA: expand KV heads to query heads.
            if num_heads != num_kv_heads:
                repeat = num_heads // num_kv_heads
                key = key.repeat_interleave(repeat, dim=0)
                value = value.repeat_interleave(repeat, dim=0)
            # query: (num_heads, 1, head_dim); single decode token attends to
            # the full history, so no causal mask is required. Compute the
            # scores/softmax in float32 (matching HF / flash-attention fp32
            # softmax accumulation) for numerical stability over long negative
            # sequences, then cast the output back to the model dtype.
            query = q[row].unsqueeze(1).float()  # (num_heads, 1, head_dim)
            key_f = key.float()
            value_f = value.float()
            scores = torch.matmul(query, key_f.transpose(-1, -2)) * scaling
            probs = torch.softmax(scores, dim=-1)
            out[row] = torch.matmul(probs, value_f).squeeze(1).to(q.dtype)

        attn_output = out.reshape(hidden.shape[0], num_heads * head_dim)
        output, _ = attn.o_proj(attn_output)
        return output

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._k_buffers.clear()
        self._v_buffers.clear()
        self._seq_lens.clear()


__all__ = ["VibeVoiceSelfManagedNegativeKVStore"]
