# SPDX-License-Identifier: Apache-2.0
"""Microsoft VibeVoice TTS serving adapter."""

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Any

import numpy as np
from vllm.multimodal.media import MediaConnector

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
)
from vllm_omni.model_executor.models.vibevoice.pipeline import (
    VIBEVOICE_VALID_TOKEN_IDS,
)
from vllm_omni.model_executor.models.vibevoice.processing_vibevoice import (
    AUDIO_BOS_TOKEN,
    AUDIO_EOS_TOKEN,
    AUDIO_TOKEN,
    MAX_AUDIO_ITEMS,
    MAX_AUDIO_SECONDS,
)
from vllm_omni.model_executor.models.vibevoice.vllm_compat import (
    get_stage0_tokenizer,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

_SYSTEM_PROMPT = (
    " Transform the text provided by various speakers into speech output, "
    "utilizing the distinct voice of each respective speaker.\n"
)
_SPEAKER_LINE = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.+)$", re.IGNORECASE)
_REFERENCE_SEGMENT = f"{AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN}"


@register_tts_adapter
class VibeVoiceTTSAdapter(ARTTSAdapter):
    """Build *ordered* reference-audio prompts for non-Realtime VibeVoice."""

    name = "vibevoice"
    stage_keys = frozenset({"vibevoice"})

    @staticmethod
    def _parse_script(text: str) -> tuple[list[tuple[int, str]], int]:
        """Return lines with speaker IDs canonicalized by first appearance."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Input text cannot be empty")

        matches = [_SPEAKER_LINE.fullmatch(line) for line in lines]
        if not any(matches):
            # Plain text is a single-speaker script. Preserve embedded newlines
            # as spaces so one input produces one deterministic speaker line.
            return [(0, " ".join(lines))], 1
        if not all(matches):
            raise ValueError(
                "VibeVoice input must be either plain text or contain only "
                "`Speaker N: text` lines; mixed formats are not supported."
            )

        speaker_map: dict[int, int] = {}
        parsed: list[tuple[int, str]] = []
        for match in matches:
            assert match is not None
            source_id = int(match.group(1))
            canonical_id = speaker_map.setdefault(source_id, len(speaker_map))
            parsed.append((canonical_id, match.group(2).strip()))
        return parsed, len(speaker_map)

    @staticmethod
    def _reference_sources(request: "OpenAICreateSpeechRequest") -> list[str]:
        ref_audio = request.ref_audio
        if ref_audio is None:
            return []
        return list(ref_audio) if isinstance(ref_audio, list) else [ref_audio]

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if request.stream:
            return "VibeVoice currently supports non-streaming speech responses only"
        try:
            _, num_speakers = self._parse_script(request.input)
        except ValueError as exc:
            return str(exc)

        references = self._reference_sources(request)
        if not references:
            return "VibeVoice requires 'ref_audio' for each speaker"
        if len(references) > MAX_AUDIO_ITEMS:
            return f"VibeVoice supports at most {MAX_AUDIO_ITEMS} reference audios per request"
        if len(references) != num_speakers:
            return (
                f"VibeVoice found {num_speakers} speakers but received "
                f"{len(references)} reference audios"
            )

        validate_format = getattr(self.ctx.server, "_validate_ref_audio_format", None)
        if callable(validate_format):
            for reference in references:
                if error := validate_format(reference):
                    return error
        if request.max_new_tokens is not None:
            if request.max_new_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"
            if request.max_new_tokens > 40500:
                return "max_new_tokens cannot exceed 40500"
        return None

    async def _resolve_reference(self, source: str) -> tuple[np.ndarray, int]:
        model_config = self.ctx.server.model_config
        connector = MediaConnector(
            allowed_local_media_path=model_config.allowed_local_media_path,
            allowed_media_domains=model_config.allowed_media_domains,
        )
        waveform, sample_rate = await connector.fetch_audio_async(source)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim not in (1, 2):
            raise ValueError(
                f"VibeVoice reference audio must be one- or two-dimensional, got {waveform.shape}."
            )
        num_samples = int(waveform.shape[0] if waveform.ndim == 1 else max(waveform.shape))
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError(f"VibeVoice reference audio sample rate must be positive, got {sample_rate}.")
        if num_samples <= 0:
            raise ValueError("VibeVoice reference audio is empty.")
        duration = num_samples / sample_rate
        if duration > MAX_AUDIO_SECONDS:
            raise ValueError(
                f"VibeVoice reference audio is {duration:.2f}s; "
                f"the maximum is {MAX_AUDIO_SECONDS}s."
            )
        return waveform, sample_rate

    @staticmethod
    def _render_prompt(parsed: list[tuple[int, str]], num_speakers: int) -> str:
        voice_prompt = " Voice input:\n" + "".join(
            f" Speaker {speaker_id}:{_REFERENCE_SEGMENT}\n"
            for speaker_id in range(num_speakers)
        )
        text_prompt = " Text input:\n" + "".join(
            f" Speaker {speaker_id}: {text}\n" for speaker_id, text in parsed
        )
        return f"{_SYSTEM_PROMPT}{voice_prompt}{text_prompt} Speech output:\n{AUDIO_BOS_TOKEN}"

    async def build(
        self,
        request: "OpenAICreateSpeechRequest",
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        parsed, num_speakers = self._parse_script(request.input)
        references = self._reference_sources(request)
        # validate() normally checks this first; keep build safe for direct use.
        if len(references) != num_speakers:
            raise ValueError(
                f"VibeVoice found {num_speakers} speakers but received "
                f"{len(references)} reference audios"
            )
        audio_items = [await self._resolve_reference(source) for source in references]
        prompt = {
            "prompt": self._render_prompt(parsed, num_speakers),
            "multi_modal_data": {"audio": audio_items},
        }
        return PreparedRequest(prompt=prompt, model_type=self.name)

    def _tokenize_prompt(self, prompt: str) -> list[int]:
        # Pinned vLLM forwards multi_modal_uuids through its token-prompt path,
        # but drops them from its text-prompt path. Supplying both text and
        # token IDs is a model-specific workaround that preserves request UUIDs
        # without patching shared vLLM/Omni runtime code.
        tokenizer = get_stage0_tokenizer(self.ctx.engine_client)
        return list(tokenizer.encode(prompt, add_special_tokens=False))

    def finalize_prepared_request(
        self,
        prepared: PreparedRequest,
        request_id: str,
    ) -> PreparedRequest:
        audio_items = prepared.prompt.get("multi_modal_data", {}).get("audio", [])
        prepared.prompt["prompt_token_ids"] = self._tokenize_prompt(
            prepared.prompt["prompt"]
        )
        prepared.prompt["multi_modal_uuids"] = {
            "audio": [f"{request_id}:audio:{item_idx}" for item_idx in range(len(audio_items))]
        }
        return prepared

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
    ) -> list:
        if not sampling_params_list:
            return sampling_params_list
        resolved = list(sampling_params_list)
        params = resolved[0]
        if isinstance(params, dict):
            params = copy.deepcopy(params)
            params.update(
                temperature=0.0,
                allowed_token_ids=list(VIBEVOICE_VALID_TOKEN_IDS),
                stop_token_ids=[151643],
                detokenize=False,
            )
        else:
            clone = getattr(params, "clone", None)
            params = clone() if callable(clone) else copy.deepcopy(params)
            params.temperature = 0.0
            params.allowed_token_ids = list(VIBEVOICE_VALID_TOKEN_IDS)
            params.stop_token_ids = [151643]
            params.detokenize = False
            if hasattr(params, "_all_stop_token_ids"):
                # Model-specific replace semantics: caller stop IDs are not
                # retained. vLLM may add the model EOS later.
                params._all_stop_token_ids = {151643}
        resolved[0] = params
        return resolved


__all__ = ["VibeVoiceTTSAdapter"]
