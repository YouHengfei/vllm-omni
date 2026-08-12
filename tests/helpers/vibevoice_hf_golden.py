# SPDX-License-Identifier: Apache-2.0
"""Isolated Transformers PR #40546 VibeVoice generation reference.

This helper intentionally runs in a fresh interpreter whose ``PYTHONPATH``
points at the requested Transformers checkout. It is test infrastructure, not a
runtime conversion or serving entry point.
"""

from __future__ import annotations

import argparse
import builtins
import json
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from safetensors import safe_open


def _deterministic_noise(step: int) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, 128, dtype=torch.float32).reshape(2, 64).add_(step * 0.03125).to(torch.bfloat16)


def _load_official_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    from transformers.models.vibevoice.convert_vibevoice_to_hf import (
        map_old_key_to_new,
    )

    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(index["weight_map"].values())):
        with safe_open(
            checkpoint / shard_name,
            framework="pt",
            device="cpu",
        ) as shard:
            for old_name in shard.keys():
                state_dict[map_old_key_to_new(old_name)] = shard.get_tensor(old_name)
    state_dict["lm_head.weight"] = state_dict["model.language_model.embed_tokens.weight"]
    return state_dict


def _install_trace(
    model: Any,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    trace: list[dict[str, torch.Tensor]] = []
    original_cfg_forward = model._run_cfg_forward
    original_sample = model._sample_audio_latent
    original_decode = model._decode_audio_latent
    semantic_encoder = model.model.semantic_tokenizer_encoder
    original_semantic_forward = semantic_encoder.forward

    def traced_cfg_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_cfg_forward(*args, **kwargs)
        condition = result[0]
        half = condition.shape[0] // 2
        trace.append(
            {
                "positive_condition": condition[:half].detach().cpu(),
                "negative_condition": condition[half:].detach().cpu(),
            }
        )
        return result

    def traced_sample(
        self: Any,
        condition: torch.Tensor,
        noise_scheduler: Any,
        num_diffusion_steps: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        if not trace:
            raise RuntimeError("VibeVoice diffusion ran before the CFG trace")
        noise = _deterministic_noise(len(trace) - 1).to(condition)
        original_randn = torch.randn

        def fixed_randn(*size: Any, **kwargs: Any) -> torch.Tensor:
            requested = tuple(size[0]) if len(size) == 1 else tuple(size)
            if requested != tuple(noise.shape):
                raise RuntimeError(
                    "Unexpected torch.randn call inside VibeVoice diffusion: "
                    f"requested={requested}, expected={tuple(noise.shape)}"
                )
            return noise.cpu()

        torch.randn = fixed_randn
        try:
            latent = original_sample(
                condition,
                noise_scheduler,
                num_diffusion_steps,
                guidance_scale,
            )
        finally:
            torch.randn = original_randn
        trace[-1]["noise"] = noise.detach().cpu()
        trace[-1]["audio_latent"] = latent.detach().cpu()
        return latent

    def traced_decode(self: Any, *args: Any, **kwargs: Any) -> Any:
        output = original_decode(*args, **kwargs)
        if not trace:
            raise RuntimeError("VibeVoice decode ran before the CFG trace")
        trace[-1]["audio"] = output.audio.detach().cpu()
        return output

    def traced_semantic_forward(*args: Any, **kwargs: Any) -> Any:
        output = original_semantic_forward(*args, **kwargs)
        if not trace:
            raise RuntimeError("VibeVoice semantic feedback ran before diffusion")
        trace[-1]["semantic_latent"] = output.latents.detach().cpu()
        return output

    model._run_cfg_forward = MethodType(traced_cfg_forward, model)
    model._sample_audio_latent = MethodType(traced_sample, model)
    model._decode_audio_latent = MethodType(traced_decode, model)
    semantic_encoder.forward = traced_semantic_forward
    return trace, {
        "sample": original_sample,
        "decode": original_decode,
        "semantic": original_semantic_forward,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--omni-trace", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    # The pinned PR checkout is missing this import in generation_vibevoice.py.
    # Injecting the documentation constant into builtins keeps the external
    # source tree untouched while exercising the PR implementation verbatim.
    from transformers.generation.logits_process import (
        LOGITS_PROCESSOR_INPUTS_DOCSTRING,
    )

    builtins.LOGITS_PROCESSOR_INPUTS_DOCSTRING = LOGITS_PROCESSOR_INPUTS_DOCSTRING
    from transformers import (
        AutoTokenizer,
        VibeVoiceConfig,
        VibeVoiceForConditionalGeneration,
    )

    config = VibeVoiceConfig.from_pretrained(
        args.runtime_schema,
        local_files_only=True,
    )
    state_dict = _load_official_state_dict(args.checkpoint)
    model, loading_info = VibeVoiceForConditionalGeneration.from_pretrained(
        None,
        config=config,
        state_dict=state_dict,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="eager",
        output_loading_info=True,
    )
    del state_dict
    model.config.text_config._attn_implementation_internal = "sdpa"
    model.model.language_model.config._attn_implementation_internal = "sdpa"
    if any(loading_info[key] for key in loading_info):
        raise RuntimeError(f"Transformers reference loading failed: {loading_info}")

    trace, original_methods = _install_trace(model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.runtime_schema,
        local_files_only=True,
    )
    prompt = "<|vision_start|><|vision_pad|><|vision_end|> Speech output:\n<|vision_start|>"
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)],
        dtype=torch.long,
        device="cuda",
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    waveform = torch.linspace(
        -0.2,
        0.2,
        3_200,
        dtype=torch.float32,
        device="cuda",
    )
    waveform *= 10 ** (-25 / 20) / (torch.sqrt(torch.mean(waveform**2)) + 1e-6)
    max_value = torch.max(torch.abs(waveform))
    if max_value > 1.0:
        waveform /= max_value + 1e-6

    original_encode = model.model.audio_tower.encode
    encoded_reference_latents: list[torch.Tensor] = []

    def encode_with_deterministic_mode(
        input_values: torch.Tensor,
        *encode_args: Any,
        **encode_kwargs: Any,
    ) -> Any:
        encode_kwargs["sample"] = False
        encoded = original_encode(
            input_values,
            *encode_args,
            **encode_kwargs,
        )
        encoded_reference_latents.append(encoded.latents.detach().cpu())
        return encoded

    model.model.audio_tower.encode = encode_with_deterministic_mode
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        input_values=waveform.to(torch.bfloat16).reshape(1, 1, 3_200),
        padding_mask=torch.ones(
            (1, 3_200),
            dtype=torch.bool,
            device="cuda",
        ),
        max_new_tokens=2,
        do_sample=False,
        guidance_scale=1.3,
        num_diffusion_steps=1,
        return_dict_in_generate=True,
    )
    token_ids = output.sequences[0, -2:].tolist()
    audio = output.audio[0]
    if audio is None:
        raise RuntimeError("Transformers VibeVoice reference produced no audio")
    if len(trace) != 2:
        raise RuntimeError(f"Transformers VibeVoice reference traced {len(trace)} steps")
    for step in trace:
        step["next_embedding"] = (
            (
                model.model.multi_modal_projector(step["audio_latent"].to("cuda"))
                + model.model.semantic_connector(step["semantic_latent"].to("cuda"))
            )
            .detach()
            .cpu()
        )

    omni_payload = torch.load(
        args.omni_trace,
        map_location="cpu",
        weights_only=True,
    )
    omni_steps = omni_payload["steps"]
    replay_trace: list[dict[str, torch.Tensor]] = []
    replay_acoustic_cache = None
    replay_semantic_cache = None
    for omni_step in omni_steps:
        positive = omni_step["positive_condition"].to("cuda")
        negative = omni_step["negative_condition"].to("cuda")
        noise = omni_step["noise"].to("cuda")
        condition = torch.cat([positive, negative], dim=0)
        replay_original_randn = torch.randn

        def replay_randn(*size: Any, **kwargs: Any) -> torch.Tensor:
            requested = tuple(size[0]) if len(size) == 1 else tuple(size)
            if requested != tuple(noise.shape):
                raise RuntimeError(
                    "Unexpected torch.randn call inside VibeVoice replay: "
                    f"requested={requested}, expected={tuple(noise.shape)}"
                )
            return noise.cpu()

        torch.randn = replay_randn
        try:
            replay_latent = original_methods["sample"](
                condition,
                model._build_default_noise_scheduler(model.generation_config),
                1,
                1.3,
            )
        finally:
            torch.randn = replay_original_randn
        replay_audio_output = original_methods["decode"](
            replay_latent,
            torch.tensor([0]),
            1,
            replay_acoustic_cache,
        )
        replay_acoustic_cache = replay_audio_output.padding_cache
        replay_semantic_output = original_methods["semantic"](
            replay_audio_output.audio,
            padding_cache=replay_semantic_cache,
            use_cache=True,
        )
        replay_semantic_cache = replay_semantic_output.padding_cache
        replay_semantic = replay_semantic_output.latents
        replay_next_embedding = model.model.multi_modal_projector(replay_latent) + model.model.semantic_connector(
            replay_semantic
        )
        replay_trace.append(
            {
                "audio_latent": replay_latent.detach().cpu(),
                "audio": replay_audio_output.audio.detach().cpu(),
                "semantic_latent": replay_semantic.detach().cpu(),
                "next_embedding": replay_next_embedding.detach().cpu(),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "token_ids": token_ids,
            "audio": audio.detach().cpu(),
            "trace": trace,
            "encoded_reference_latents": encoded_reference_latents,
            "replay_trace": replay_trace,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
