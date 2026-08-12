from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.multimodal.media import MediaConnector

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters import resolve_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.vibevoice import VibeVoiceTTSAdapter
from vllm_omni.model_executor.models.vibevoice.pipeline import VIBEVOICE_VALID_TOKEN_IDS
from vllm_omni.model_executor.models.vibevoice.processing_vibevoice import (
    AUDIO_BOS_TOKEN,
    AUDIO_EOS_TOKEN,
    AUDIO_TOKEN,
    MAX_AUDIO_ITEMS,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _adapter() -> VibeVoiceTTSAdapter:
    server = SimpleNamespace(
        _validate_ref_audio_format=lambda _: None,
        model_config=SimpleNamespace(
            allowed_local_media_path=None,
            allowed_media_domains=None,
        ),
    )
    tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=False: list(text.encode("utf-8"))
    )
    engine_client = SimpleNamespace(
        engine=SimpleNamespace(
            input_processor=SimpleNamespace(
                renderer=SimpleNamespace(get_tokenizer=lambda: tokenizer)
            )
        )
    )
    return VibeVoiceTTSAdapter(
        SpeechServingContext(server=server, engine_client=engine_client)
    )


def test_vibevoice_adapter_is_registered_and_detected() -> None:
    assert resolve_adapter("vibevoice") is VibeVoiceTTSAdapter

    serving = object.__new__(OmniOpenAIServingSpeech)
    serving._tts_stage = SimpleNamespace(
        engine_args=SimpleNamespace(
            model_stage="vibevoice",
            model_arch="VibeVoiceForConditionalGeneration",
        )
    )
    assert serving._detect_tts_model_type() == "vibevoice"


def test_vibevoice_adapter_rejects_mismatched_or_ambiguous_speakers() -> None:
    adapter = _adapter()
    mismatch = OpenAICreateSpeechRequest(
        input="Speaker 1: hello\nSpeaker 2: world",
        ref_audio=["file:///one.wav"],
    )
    assert adapter.validate(mismatch) == (
        "VibeVoice found 2 speakers but received 1 reference audios"
    )

    mixed = OpenAICreateSpeechRequest(
        input="Speaker 1: hello\nthis line has no speaker",
        ref_audio=["file:///one.wav"],
    )
    assert "mixed formats" in (adapter.validate(mixed) or "")

    too_many = OpenAICreateSpeechRequest(
        input="\n".join(f"Speaker {idx}: text" for idx in range(MAX_AUDIO_ITEMS + 1)),
        ref_audio=[f"file:///{idx}.wav" for idx in range(MAX_AUDIO_ITEMS + 1)],
    )
    assert f"at most {MAX_AUDIO_ITEMS}" in (adapter.validate(too_many) or "")


def test_vibevoice_adapter_rejects_streaming() -> None:
    request = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="file:///voice.wav",
        stream=True,
    )
    assert "non-streaming" in (_adapter().validate(request) or "")


@pytest.mark.parametrize(
    ("waveform", "sample_rate", "message"),
    [
        (np.zeros((2, 2, 2), dtype=np.float32), 24_000, "one- or two-dimensional"),
        (np.zeros(0, dtype=np.float32), 24_000, "reference audio is empty"),
        (np.zeros(16, dtype=np.float32), 0, "sample rate must be positive"),
        (np.zeros(60 * 24_000 + 1, dtype=np.float32), 24_000, "maximum is 60s"),
    ],
)
def test_vibevoice_adapter_reference_media_errors_are_request_visible(
    monkeypatch: pytest.MonkeyPatch,
    waveform: np.ndarray,
    sample_rate: int,
    message: str,
) -> None:
    adapter = _adapter()

    async def fetch_audio_async(_self, _source):
        return waveform, sample_rate

    monkeypatch.setattr(MediaConnector, "fetch_audio_async", fetch_audio_async)

    with pytest.raises(ValueError, match=message):
        asyncio.run(adapter._resolve_reference("file:///bad.wav"))


def test_vibevoice_adapter_builds_ordered_prompt_audio_and_request_uuids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter()
    request = OpenAICreateSpeechRequest(
        input="Speaker 8: first\nSpeaker 3: second\nSpeaker 8: third",
        ref_audio=["ref-a", "ref-b"],
    )

    async def resolve(source: str):
        value = 1.0 if source == "ref-a" else 2.0
        return np.full(3200, value, dtype=np.float32), 24_000

    monkeypatch.setattr(adapter, "_resolve_reference", resolve)
    assert adapter.validate(request) is None
    prepared = asyncio.run(adapter.build(request, [], True))
    prepared = adapter.finalize_prepared_request(prepared, "speech-request-7")

    prompt = prepared.prompt["prompt"]
    reference_segment = f"{AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN}"
    assert prompt.count(reference_segment) == 2
    assert " Speaker 0: first\n" in prompt
    assert " Speaker 1: second\n" in prompt
    assert " Speaker 0: third\n" in prompt
    assert prompt.endswith(f" Speech output:\n{AUDIO_BOS_TOKEN}")

    audio_items = prepared.prompt["multi_modal_data"]["audio"]
    assert len(audio_items) == 2
    assert all(isinstance(item, tuple) and item[1] == 24_000 for item in audio_items)
    np.testing.assert_array_equal(audio_items[0][0], np.full(3200, 1.0, dtype=np.float32))
    np.testing.assert_array_equal(audio_items[1][0], np.full(3200, 2.0, dtype=np.float32))
    assert prepared.prompt["prompt_token_ids"] == list(prompt.encode("utf-8"))
    assert prepared.prompt["multi_modal_uuids"] == {
        "audio": [
            "speech-request-7:audio:0",
            "speech-request-7:audio:1",
        ]
    }
    assert "hf_processor_mm_kwargs" not in prepared.prompt


def test_normal_speech_serving_path_applies_uuid_and_sampling_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=False: list(text.encode("utf-8"))
    )

    def generate(**kwargs):
        captured.update(kwargs)
        return "generator"

    engine_client = SimpleNamespace(
        errored=False,
        default_sampling_params_list=[
            SamplingParams(
                allowed_token_ids=[1],
                stop_token_ids=[2],
                detokenize=True,
            )
        ],
        model_config=SimpleNamespace(async_chunk=False),
        engine=SimpleNamespace(
            input_processor=SimpleNamespace(
                renderer=SimpleNamespace(get_tokenizer=lambda: tokenizer)
            )
        ),
        generate=generate,
    )
    serving = object.__new__(OmniOpenAIServingSpeech)
    serving.engine_client = engine_client
    serving.model_config = engine_client.model_config
    serving._tts_model_type = "vibevoice"
    serving._is_tts = True
    serving._adapter = VibeVoiceTTSAdapter(
        SpeechServingContext(server=serving, engine_client=engine_client)
    )
    serving._track_ref_audio_artifact_warmup = lambda *args, **kwargs: None
    serving._validate_ref_audio_format = lambda _: None

    async def resolve(_source: str):
        return np.zeros(3200, dtype=np.float32), 24_000

    monkeypatch.setattr(serving._adapter, "_resolve_reference", resolve)
    request = OpenAICreateSpeechRequest(input="hello", ref_audio="ref-a")

    request_id, generator, _ = asyncio.run(
        serving._prepare_speech_generation(request, request_id="speech-fixed")
    )

    assert request_id == "speech-fixed"
    assert generator == "generator"
    assert captured["prompt"]["multi_modal_uuids"] == {
        "audio": ["speech-fixed:audio:0"]
    }
    assert captured["prompt"]["prompt_token_ids"]
    params = captured["sampling_params_list"][0]
    assert params.temperature == 0.0
    assert params.allowed_token_ids == VIBEVOICE_VALID_TOKEN_IDS
    assert params.stop_token_ids == [151643]
    assert params.all_stop_token_ids == {151643}
    assert params.detokenize is False


def test_vibevoice_sampling_constraints_replace_object_without_mutating_caller() -> None:
    adapter = _adapter()
    request = OpenAICreateSpeechRequest(input="hello", ref_audio="ref")
    caller = SamplingParams(
        temperature=0.7,
        max_tokens=123,
        allowed_token_ids=[1],
        stop_token_ids=[2],
        detokenize=True,
    )
    caller.skip_clone = True
    caller_before = copy.deepcopy(caller)

    (resolved,) = adapter.apply_sampling_overrides([caller], request)

    assert resolved is not caller
    assert resolved.temperature == 0.0
    assert resolved.max_tokens == 123
    assert resolved.allowed_token_ids == VIBEVOICE_VALID_TOKEN_IDS
    assert resolved.stop_token_ids == [151643]
    assert resolved.all_stop_token_ids == {151643}
    assert resolved.detokenize is False
    assert caller == caller_before


def test_vibevoice_sampling_constraints_force_official_argmax() -> None:
    adapter = _adapter()
    request = OpenAICreateSpeechRequest(input="hello", ref_audio="ref")
    caller = SamplingParams(
        temperature=0.8,
        allowed_token_ids=[1],
        stop_token_ids=[2],
        detokenize=True,
    )

    (resolved,) = adapter.apply_sampling_overrides([caller], request)

    # Microsoft generation always selects the control token with argmax even
    # when do_sample=True. The serving normalization must therefore prevent a
    # caller temperature from turning the four-token gate into random sampling.
    assert resolved.temperature == 0.0


def test_vibevoice_nonstreaming_serving_serializes_published_waveform() -> None:
    serving = object.__new__(OmniOpenAIServingSpeech)
    serving._tts_model_type = "vibevoice"
    serving._request_ref_audio_artifact_keys = {}
    serving._ref_audio_model_artifact_ready = set()
    serving._ref_audio_resolve_cache = {}
    waveform = np.linspace(-0.25, 0.25, 3_200, dtype=np.float32)
    multimodal_output = {
        "audio": torch.from_numpy(waveform),
        "sr": torch.tensor(24_000, dtype=torch.int32),
    }
    final = SimpleNamespace(
        multimodal_output=multimodal_output,
        request_output=None,
        outputs=[],
    )

    async def generator():
        yield final

    async def prepare(*_args, **_kwargs):
        return "vibevoice-serving", generator(), {}

    serving._prepare_speech_generation = prepare
    request = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="ref",
        response_format="wav",
    )

    audio_bytes, media_type = asyncio.run(serving._generate_audio_bytes(request))

    assert media_type == "audio/wav"
    assert isinstance(audio_bytes, bytes)
    assert audio_bytes[:4] == b"RIFF"
    assert b"WAVE" in audio_bytes[:16]
    assert len(audio_bytes) > 44


def test_vibevoice_sampling_constraints_replace_dict_and_are_idempotent() -> None:
    adapter = _adapter()
    request = OpenAICreateSpeechRequest(input="hello", ref_audio="ref")
    caller = {
        "temperature": 0.5,
        "allowed_token_ids": [9],
        "stop_token_ids": [8],
        "detokenize": True,
    }
    caller_before = copy.deepcopy(caller)

    first = adapter.apply_sampling_overrides([caller], request)
    second = adapter.apply_sampling_overrides(first, request)

    assert first == second
    assert first[0] == {
        "temperature": 0.0,
        "allowed_token_ids": VIBEVOICE_VALID_TOKEN_IDS,
        "stop_token_ids": [151643],
        "detokenize": False,
    }
    assert caller == caller_before
