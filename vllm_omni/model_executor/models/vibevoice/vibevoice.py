# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Weight-name mappings are adapted from Hugging Face Transformers PR #40546,
# src/transformers/models/vibevoice/convert_vibevoice_to_hf.py.
# Copyright 2026 The HuggingFace Inc. team. Licensed under Apache-2.0.
"""VibeVoice model implementation helpers.

The model class will live in this module. Keep checkpoint compatibility next to
that class so the model remains the owner of its loading semantics.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .audio_decode import (
    VibeVoiceAudioTokenDecodeOutput,
    VibeVoiceAudioTokenDecoder,
)
from .diffusion import (
    VibeVoiceDiffusionHead,
    VibeVoiceDiffusionSampler,
    VibeVoiceRMSNorm,
)
from .processing_vibevoice import (
    AUDIO_BOS_TOKEN,
    AUDIO_EOS_TOKEN,
    AUDIO_HOP_LENGTH,
    AUDIO_TOKEN,
    VibeVoiceDummyInputsBuilder,
    VibeVoiceMultiModalProcessor,
    VibeVoiceProcessingInfo,
)
from .stateful import (
    VibeVoiceNegativeKVBranch,
    VibeVoiceStatefulInference,
)
from .vllm_compat import merge_multimodal_embeddings


def _num_tokenizer_stages(config: Any, child_config_name: str) -> int:
    """Return the tokenizer stage count needed for N -> N-1 key mappings."""
    child_config = getattr(config, child_config_name, None)
    depths = getattr(child_config, "depths", None)
    if not isinstance(depths, (list, tuple)) or not depths:
        raise ValueError(
            f"VibeVoice {child_config_name}.depths must be a non-empty list or tuple, "
            f"got {depths!r}."
        )
    return len(depths)


def _add_encoder_mappings(
    mappings: dict[re.Pattern[str], str | None],
    *,
    source: str,
    target: str,
    num_stages: int,
) -> None:
    """Add Microsoft tokenizer-encoder to HF Acoustic Encoder mappings."""
    mappings[re.compile(rf"{source}\.encoder\.downsample_layers\.0\.0\.conv\.")] = (
        f"{target}.stem.conv.conv."
    )
    mappings[re.compile(rf"{source}\.encoder\.stages\.0\.")] = f"{target}.stem.stage."

    for source_idx in range(1, num_stages):
        target_idx = source_idx - 1
        mappings[
            re.compile(rf"{source}\.encoder\.downsample_layers\.{source_idx}\.0\.conv\.")
        ] = f"{target}.conv_layers.{target_idx}.conv.conv."
        mappings[re.compile(rf"{source}\.encoder\.stages\.{source_idx}\.")] = (
            f"{target}.conv_layers.{target_idx}.stage."
        )

    mappings[re.compile(rf"{source}\.encoder\.head\.conv\.")] = f"{target}.head."


def _add_decoder_mappings(
    mappings: dict[re.Pattern[str], str | None],
    *,
    source: str,
    target: str,
    num_stages: int,
) -> None:
    """Add Microsoft tokenizer-decoder to HF Acoustic Decoder mappings."""
    mappings[
        re.compile(rf"{source}\.decoder\.upsample_layers\.0\.0\.conv\.conv\.")
    ] = f"{target}.stem.conv.conv."
    mappings[re.compile(rf"{source}\.decoder\.stages\.0\.")] = f"{target}.stem.stage."

    for source_idx in range(1, num_stages):
        target_idx = source_idx - 1
        mappings[
            re.compile(
                rf"{source}\.decoder\.upsample_layers\.{source_idx}\.0\.convtr\.convtr\."
            )
        ] = f"{target}.conv_layers.{target_idx}.convtr.convtr."
        mappings[re.compile(rf"{source}\.decoder\.stages\.{source_idx}\.")] = (
            f"{target}.conv_layers.{target_idx}.stage."
        )

    mappings[re.compile(rf"{source}\.decoder\.head\.conv\.")] = f"{target}.head."


def _build_vibevoice_weights_mapper(config: Any) -> WeightsMapper:
    """Build the Microsoft-checkpoint to HF-runtime name mapper.

    The mapper is also safe for checkpoints already converted to the PR #40546
    layout: none of the source patterns match those names, so they pass through
    unchanged.

    A builder is needed because ``WeightsMapper`` regex replacements cannot
    express the tokenizer's ``source_index - 1`` transformation. We generate
    exact regex entries from the normalized child configs, while
    ``WeightsMapper`` still performs every runtime key conversion.
    """
    acoustic_stages = _num_tokenizer_stages(config, "audio_config")
    semantic_stages = _num_tokenizer_stages(config, "semantic_model_config")

    # dict preserves insertion order. Ordering is significant: specific
    # tokenizer/diffusion mappings must run before their generic cleanup rules.
    mappings: dict[re.Pattern[str], str | None] = {}

    _add_encoder_mappings(
        mappings,
        source=r"semantic_tokenizer",
        target="semantic_tokenizer_encoder",
        num_stages=semantic_stages,
    )
    _add_encoder_mappings(
        mappings,
        source=r"acoustic_tokenizer",
        target="audio_tower.encoder",
        num_stages=acoustic_stages,
    )
    _add_decoder_mappings(
        mappings,
        source=r"acoustic_tokenizer",
        target="audio_tower.decoder",
        num_stages=acoustic_stages,
    )

    mappings.update(
        {
            # Any remaining Acoustic Tokenizer keys belong below audio_tower.
            re.compile(r"acoustic_tokenizer\."): "audio_tower.",
            # Diffusion Head.
            re.compile(r"prediction_head\.t_embedder\.mlp\.0\."): (
                "diffusion_head.timestep_proj.layer_1."
            ),
            re.compile(r"prediction_head\.t_embedder\.mlp\.2\."): (
                "diffusion_head.timestep_proj.layer_2."
            ),
            re.compile(r"prediction_head\.layers\.(\d+)\.adaLN_modulation\.1\."): (
                r"diffusion_head.layers.\1.linear."
            ),
            re.compile(r"prediction_head\.final_layer\.adaLN_modulation\.1\."): (
                "diffusion_head.final_layer.linear_1."
            ),
            re.compile(r"prediction_head\.final_layer\.linear\."): (
                "diffusion_head.final_layer.linear_2."
            ),
            re.compile(r"prediction_head\."): "diffusion_head.",
            # Acoustic and semantic connectors.
            re.compile(r"acoustic_connector\.fc1\."): "multi_modal_projector.linear_1.",
            re.compile(r"acoustic_connector\.norm\."): "multi_modal_projector.act.",
            re.compile(r"acoustic_connector\.fc2\."): "multi_modal_projector.linear_2.",
            re.compile(r"semantic_connector\.fc1\."): "semantic_connector.linear_1.",
            re.compile(r"semantic_connector\.norm\."): "semantic_connector.act.",
            re.compile(r"semantic_connector\.fc2\."): "semantic_connector.linear_2.",
            # Latent normalization factors.
            re.compile(r"^model\.speech_scaling_factor$"): "model.latent_scaling_factor",
            re.compile(r"^model\.speech_bias_factor$"): "model.latent_bias_factor",
            # Original modules contain one extra nested Conv1d wrapper.
            re.compile(r"mixer\.conv\.conv\.conv\."): "mixer.conv.",
            re.compile(r"\.conv\.conv\.conv\."): ".conv.conv.",
        }
    )

    return WeightsMapper(orig_to_new_regex=mappings)


class VibeVoiceMultiModalProjector(nn.Module):
    """Project a continuous Acoustic/Semantic latent into Qwen2 hidden space."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, output_dim)
        self.act = VibeVoiceRMSNorm(output_dim)
        self.linear_2 = nn.Linear(output_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(features)))


class VibeVoiceModel(nn.Module):
    """Weight-complete VibeVoice backbone scaffold.

    Forward-time multimodal replacement and per-request decoder state are added
    separately; this class already mirrors the released checkpoint hierarchy.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.audio_tower = AutoModel.from_config(config.audio_config)
        self.language_model = Qwen2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.multi_modal_projector = VibeVoiceMultiModalProjector(
            config.audio_config.hidden_size, config.text_config.hidden_size
        )
        self.semantic_tokenizer_encoder = AutoModel.from_config(config.semantic_model_config)
        self.semantic_connector = VibeVoiceMultiModalProjector(
            config.semantic_model_config.hidden_size, config.text_config.hidden_size
        )
        self.diffusion_head = VibeVoiceDiffusionHead(config)
        # Pure model-side numerical helper. It creates a fresh DPM solver for
        # each audio token and owns no request/KV/cache state.
        self.diffusion_sampler = VibeVoiceDiffusionSampler.from_model_config(config)
        # Like the diffusion sampler, this kernel receives and returns caches;
        # it never owns mutable request state.
        self.audio_token_decoder = VibeVoiceAudioTokenDecoder.from_model_config(config)
        self.latent_scaling_factor = nn.Parameter(torch.tensor(1.0))
        self.latent_bias_factor = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def sample_audio_latent(
        self,
        positive_condition: torch.Tensor,
        negative_condition: torch.Tensor,
        noise: torch.Tensor,
        *,
        guidance_scale: float,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Run the model-local diffusion numerical kernel for one AR step."""
        return self.diffusion_sampler.sample_audio_latent(
            self.diffusion_head,
            positive_condition,
            negative_condition,
            noise,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

    def decode_audio_token(
        self,
        audio_latent: torch.Tensor,
        *,
        acoustic_cache: Any = None,
        semantic_cache: Any = None,
    ) -> VibeVoiceAudioTokenDecodeOutput:
        """Decode one acoustic latent and produce semantic AR feedback."""
        return self.audio_token_decoder.decode_audio_token(
            audio_tower=self.audio_tower,
            semantic_encoder=self.semantic_tokenizer_encoder,
            acoustic_projector=self.multi_modal_projector,
            semantic_connector=self.semantic_connector,
            latent_scaling_factor=self.latent_scaling_factor,
            latent_bias_factor=self.latent_bias_factor,
            audio_latent=audio_latent,
            acoustic_cache=acoustic_cache,
            semantic_cache=semantic_cache,
        )


@MULTIMODAL_REGISTRY.register_processor(
    VibeVoiceMultiModalProcessor,
    info=VibeVoiceProcessingInfo,
    dummy_inputs=VibeVoiceDummyInputsBuilder,
)
class VibeVoiceForConditionalGeneration(nn.Module, SupportsMultiModal):
    """vLLM VibeVoice model with reference-audio prefill support."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "audio":
            return f"{AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN}"
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.has_preprocess = True
        self.has_postprocess = True
        # Only the final scheduled row is needed as the positive diffusion
        # condition; never reconstruct a full prefix-cache hidden span.
        self.requires_full_prefix_cached_hidden_states = False
        self.postprocess_uses_multimodal_outputs = False
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model = VibeVoiceModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        # The released checkpoint ties the output head to the token embedding
        # and consequently contains no independent lm_head tensor.
        self.lm_head = self.model.language_model.embed_tokens
        self.logits_processor = LogitsProcessor(self.config.text_config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.language_model.make_empty_intermediate_tensors

        self._stateful = VibeVoiceStatefulInference(
            audio_bos_token_id=int(self.config.audio_bos_token_id),
            audio_eos_token_id=int(self.config.audio_eos_token_id),
            audio_token_id=int(self.config.audio_token_id),
            eos_token_id=int(self.config.eos_token_id),
            latent_size=int(self.config.audio_config.hidden_size),
            condition_size=int(self.config.text_config.hidden_size),
            default_guidance_scale=1.3,
            default_num_diffusion_steps=(
                self.model.diffusion_sampler.default_num_inference_steps
            ),
        )
        self._negative_kv_branch: VibeVoiceNegativeKVBranch | None = None
        self._pending_request_ids: list[str] = []
        self._pending_request_spans: list[tuple[str, int, int]] = []
        self._pending_audio_transitions: list[tuple[str, int]] = []
        self._pending_num_input_rows = 0

    def get_language_model(self) -> Qwen2Model:
        return self.model.language_model

    def get_input_embeddings(self) -> nn.Module:
        return self.model.language_model.embed_tokens

    def _get_audio_embeddings(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor,
        audio_num_tokens: torch.Tensor | None = None,
        *,
        sample: bool,
    ) -> list[torch.Tensor]:
        """Encode, project, and crop every reference-audio item.

        ``sample`` is explicit so tests can exercise a deterministic parity
        path. Runtime ``embed_multimodal`` always uses the official
        ``sample=True`` behavior.
        """
        if input_values.ndim != 3:
            raise ValueError(
                "VibeVoice input_values must have shape "
                f"(batch, channels, samples), got {tuple(input_values.shape)}."
            )
        if padding_mask.ndim == 1:
            padding_mask = padding_mask.unsqueeze(0)
        if padding_mask.ndim != 2:
            raise ValueError(
                "VibeVoice padding_mask must have shape (batch, samples), "
                f"got {tuple(padding_mask.shape)}."
            )
        if input_values.shape[0] != padding_mask.shape[0]:
            raise ValueError(
                "VibeVoice audio batch mismatch: "
                f"input_values={input_values.shape[0]}, "
                f"padding_mask={padding_mask.shape[0]}."
            )
        if input_values.shape[1] != 1:
            raise ValueError(
                "VibeVoice Acoustic Encoder requires mono input, got "
                f"{input_values.shape[1]} channels."
            )
        if input_values.shape[-1] != padding_mask.shape[-1]:
            raise ValueError(
                "VibeVoice waveform/mask length mismatch: "
                f"input_values={input_values.shape[-1]}, "
                f"padding_mask={padding_mask.shape[-1]}."
            )

        tower_param = next(self.model.audio_tower.parameters())
        input_values = input_values.to(
            device=tower_param.device,
            dtype=tower_param.dtype,
        )
        padding_mask = padding_mask.to(device=tower_param.device)
        counts_from_mask = torch.div(
            padding_mask.to(torch.long).sum(dim=-1) + AUDIO_HOP_LENGTH - 1,
            AUDIO_HOP_LENGTH,
            rounding_mode="floor",
        )
        if audio_num_tokens is None:
            audio_num_tokens = counts_from_mask
        else:
            audio_num_tokens = torch.as_tensor(
                audio_num_tokens,
                device=counts_from_mask.device,
                dtype=torch.long,
            ).reshape(-1)
            if audio_num_tokens.shape != counts_from_mask.shape or not torch.equal(
                audio_num_tokens,
                counts_from_mask,
            ):
                raise ValueError(
                    "VibeVoice audio_num_tokens does not match padding_mask: "
                    f"provided={audio_num_tokens.tolist()}, "
                    f"expected={counts_from_mask.tolist()}."
                )

        with torch.no_grad():
            acoustic_latents = self.model.audio_tower.encode(
                input_values,
                sample=sample,
            ).latents
            acoustic_features = (
                acoustic_latents
                + self.model.latent_bias_factor.to(acoustic_latents.device)
            ) * self.model.latent_scaling_factor.to(acoustic_latents.device)
            projected = self.model.multi_modal_projector(acoustic_features)

        if projected.ndim != 3 or projected.shape[0] != input_values.shape[0]:
            raise ValueError(
                "VibeVoice Acoustic Encoder returned an unexpected shape: "
                f"{tuple(projected.shape)}."
            )

        embeddings: list[torch.Tensor] = []
        for item_idx, num_tokens in enumerate(audio_num_tokens.tolist()):
            if num_tokens < 1 or num_tokens > projected.shape[1]:
                raise ValueError(
                    f"VibeVoice audio item {item_idx} requires {num_tokens} "
                    f"embeddings, but the encoder produced {projected.shape[1]}."
                )
            item = projected[item_idx, :num_tokens]
            if item.shape[0] != num_tokens:
                raise AssertionError(
                    "VibeVoice audio embedding/placeholder length mismatch: "
                    f"item={item_idx}, embeddings={item.shape[0]}, "
                    f"placeholders={num_tokens}."
                )
            embeddings.append(item)
        return embeddings

    def embed_multimodal(self, **kwargs: object) -> list[torch.Tensor]:
        input_values = kwargs.get("input_values")
        padding_mask = kwargs.get("padding_mask")
        if not isinstance(input_values, torch.Tensor):
            raise TypeError("VibeVoice embed_multimodal requires tensor input_values.")
        if not isinstance(padding_mask, torch.Tensor):
            raise TypeError("VibeVoice embed_multimodal requires tensor padding_mask.")
        audio_num_tokens = kwargs.get("audio_num_tokens")
        if audio_num_tokens is not None and not isinstance(
            audio_num_tokens, torch.Tensor
        ):
            audio_num_tokens = torch.as_tensor(audio_num_tokens)
        return self._get_audio_embeddings(
            input_values,
            padding_mask,
            audio_num_tokens,
            sample=True,
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.language_model.embed_input_ids(input_ids)
        if multimodal_embeddings is None or is_multimodal is None:
            return inputs_embeds
        return merge_multimodal_embeddings(
            inputs_embeds,
            multimodal_embeddings,
            is_multimodal,
        )

    def bind_negative_kv_branch(
        self,
        branch: VibeVoiceNegativeKVBranch,
    ) -> None:
        """Bind the future independent negative-Qwen PagedAttention owner."""
        self._negative_kv_branch = branch

    def record_negative_condition(
        self,
        request_id: str,
        condition: torch.Tensor,
    ) -> None:
        """Publish one aligned hidden row from the negative Qwen branch."""
        self._stateful.record_negative_condition(request_id, condition)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Apply control-token transitions and continuous feedback embeddings."""
        request_id = info_dict.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("VibeVoice preprocess requires a non-empty request_id.")
        self._stateful.flush_deferred_cleanup(
            exclude_request_ids={request_id},
        )

        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        is_prefill = bool(info_dict.get("_omni_is_prefill", input_ids.numel() > 1))
        num_computed = int(info_dict.get("_omni_num_computed_tokens", 0) or 0)
        prompt_len = int(info_dict.get("_omni_prompt_len", input_ids.numel()) or 0)
        state = self._stateful.get_or_create(
            request_id,
            reset=is_prefill and num_computed == 0,
        )

        if is_prefill:
            # Serving prompts already end in audio BOS. Initialize the segment
            # at the final prefill chunk so the first sampled audio token is a
            # valid transition, matching Transformers generation.
            is_final_prefill = num_computed + int(input_ids.numel()) >= prompt_len
            if (
                is_final_prefill
                and input_ids.numel() > 0
                and int(input_ids[-1].item()) == self._stateful.audio_bos_token_id
            ):
                self._stateful.start_audio_segment(
                    state.request_id,
                    self._negative_kv_branch,
                )
        elif input_ids.numel() == 1:
            token_id = int(input_ids.reshape(-1)[0].item())
            if token_id == self._stateful.audio_token_id:
                # Defer M4a so all active audio-token requests in this runner
                # step consume one official [2B, latent] RNG draw.
                self._pending_audio_transitions.append(
                    (request_id, self._pending_num_input_rows)
                )
            else:
                next_embedding, _ = self._stateful.process_sampled_token(
                    request_id=request_id,
                    token_id=token_id,
                    token_embedding=input_embeds.reshape(1, -1),
                    kernel=self.model,
                    negative_kv_branch=self._negative_kv_branch,
                )
                input_embeds = next_embedding

        span_start = self._pending_num_input_rows
        span_end = span_start + int(input_ids.numel())
        self._pending_request_ids.append(request_id)
        self._pending_request_spans.append((request_id, span_start, span_end))
        self._pending_num_input_rows = span_end
        return input_ids, input_embeds, {"request_id": request_id}

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        **info_dict: Any,
    ) -> dict[str, Any]:
        """Retain only the positive hidden row needed by the next transition."""
        request_id = info_dict.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return {}
        if hidden_states.numel() > 0:
            condition = hidden_states[-1].detach().reshape(1, -1).contiguous()
            self._stateful.record_positive_condition(request_id, condition)
        self._stateful.finish_postprocess(request_id)
        return {"request_id": request_id}

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        self._stateful.on_requests_finished(finished_req_ids)
        if self._negative_kv_branch is not None:
            self._negative_kv_branch.on_requests_finished(finished_req_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        sampling_extra_args: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        pending_request_ids = self._pending_request_ids
        pending_request_spans = self._pending_request_spans
        pending_audio_transitions = self._pending_audio_transitions
        self._pending_request_ids = []
        self._pending_request_spans = []
        self._pending_audio_transitions = []
        self._pending_num_input_rows = 0
        if sampling_extra_args:
            for request_id, extra_args in zip(
                pending_request_ids,
                sampling_extra_args,
                strict=False,
            ):
                self._stateful.set_runtime_controls(request_id, extra_args)

        if pending_audio_transitions:
            if inputs_embeds is None:
                raise RuntimeError(
                    "VibeVoice audio-token feedback requires inputs_embeds."
                )
            if self._negative_kv_branch is not None:
                negative_request_ids = [
                    request_id for request_id, _ in pending_audio_transitions
                ]
                negative_inputs: list[torch.Tensor] = []
                for request_id in negative_request_ids:
                    state = self._stateful.get(request_id)
                    negative_input = (
                        state.negative_input_embedding
                        if state is not None
                        else None
                    )
                    if negative_input is None:
                        raise RuntimeError(
                            "VibeVoice negative Qwen branch has no preceding "
                            f"input embedding for request {request_id!r}."
                        )
                    negative_inputs.append(negative_input)
                negative_conditions = self._negative_kv_branch.forward_step(
                    negative_request_ids,
                    negative_inputs,
                )
                if len(negative_conditions) != len(negative_request_ids):
                    raise RuntimeError(
                        "VibeVoice negative Qwen branch returned a condition "
                        "batch with the wrong length."
                    )
                for request_id, condition in zip(
                    negative_request_ids,
                    negative_conditions,
                    strict=True,
                ):
                    self._stateful.record_negative_condition(
                        request_id,
                        condition,
                    )

            # Different request controls imply different DPM loops. Group by
            # loop contract while preserving first-seen request order.
            transition_groups: dict[
                tuple[float, int],
                list[tuple[str, int]],
            ] = {}
            for request_id, row_offset in pending_audio_transitions:
                state = self._stateful.get(request_id)
                if state is None:
                    raise RuntimeError(
                        f"Missing VibeVoice request state for {request_id!r}."
                    )
                transition_groups.setdefault(
                    (state.guidance_scale, state.num_diffusion_steps),
                    [],
                ).append((request_id, row_offset))

            for transitions in transition_groups.values():
                request_ids = [item[0] for item in transitions]
                offsets = [item[1] for item in transitions]
                token_embeddings = [
                    inputs_embeds[offset : offset + 1]
                    for offset in offsets
                ]
                next_embeddings, _ = self._stateful.process_audio_tokens_batch(
                    request_ids=request_ids,
                    token_embeddings=token_embeddings,
                    kernel=self.model,
                )
                offset_tensor = torch.tensor(
                    offsets,
                    device=inputs_embeds.device,
                    dtype=torch.long,
                )
                inputs_embeds.index_copy_(
                    0,
                    offset_tensor,
                    torch.cat(next_embeddings, dim=0).to(inputs_embeds),
                )

        # Save the exact embedding consumed by the current positive Qwen step.
        # If that step samples audio_token, the future negative branch advances
        # this embedding before M4a on the next runner iteration.
        if inputs_embeds is not None:
            for request_id, span_start, span_end in pending_request_spans:
                state = self._stateful.get(request_id)
                if state is not None and state.in_audio_segment and span_end > span_start:
                    self._stateful.record_negative_input_embedding(
                        request_id,
                        inputs_embeds[span_end - 1 : span_end],
                    )

        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = _build_vibevoice_weights_mapper(self.config)
        return AutoWeightsLoader(self).load_weights(weights, mapper=mapper)


__all__ = [
    "VibeVoiceAudioTokenDecodeOutput",
    "VibeVoiceAudioTokenDecoder",
    "VibeVoiceDiffusionHead",
    "VibeVoiceForConditionalGeneration",
    "VibeVoiceModel",
    "VibeVoiceMultiModalProjector",
    "VibeVoiceRMSNorm",
    "_build_vibevoice_weights_mapper",
]
