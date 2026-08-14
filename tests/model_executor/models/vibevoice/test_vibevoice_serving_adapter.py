from __future__ import annotations

import asyncio
import base64
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.multimodal.media import MediaConnector

from vllm_omni.entrypoints.openai.protocol.audio import (
    BatchSpeechRequest,
    CreateAudio,
    OpenAICreateSpeechRequest,
    SpeechBatchItem,
    SpeechTokenUsage,
)
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters import resolve_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.vibevoice import VibeVoiceTTSAdapter
from vllm_omni.model_executor.models.vibevoice.pipeline import VIBEVOICE_VALID_TOKEN_IDS
from vllm_omni.model_executor.models.vibevoice.processing_vibevoice import (
    AUDIO_BOS_TOKEN,
    AUDIO_EOS_TOKEN,
    AUDIO_TOKEN,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _adapter() -> VibeVoiceTTSAdapter:
    server = SimpleNamespace(
        _validate_ref_audio_format=lambda _: None,
        uploaded_speakers={},
        model_config=SimpleNamespace(
            allowed_local_media_path=None,
            allowed_media_domains=None,
        ),
    )
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: list(text.encode("utf-8")))
    engine_client = SimpleNamespace(
        engine=SimpleNamespace(
            input_processor=SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda: tokenizer))
        )
    )
    return VibeVoiceTTSAdapter(SpeechServingContext(server=server, engine_client=engine_client))


def _uploaded_voice_adapter(
    *,
    voice_name: str = "alice",
    embedding_source: str = "audio",
    audio_data: str | None = "data:audio/wav;base64,dGVzdA==",
    ref_text: str | None = "stored transcript",
) -> VibeVoiceTTSAdapter:
    adapter = _adapter()
    server = adapter.ctx.server
    server._tts_model_type = "vibevoice"
    server.uploaded_speakers = {
        voice_name.lower(): {
            "name": voice_name,
            "embedding_source": embedding_source,
            "ref_text": ref_text,
        }
    }
    server._get_uploaded_audio_data = lambda _voice: audio_data
    server._apply_uploaded_speaker = lambda request: OmniOpenAIServingSpeech._apply_uploaded_speaker(server, request)
    return adapter


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
    assert adapter.validate(mismatch) == ("VibeVoice found 2 speakers but received 1 reference audios")

    mixed = OpenAICreateSpeechRequest(
        input="Speaker 1: hello\nthis line has no speaker",
        ref_audio=["file:///one.wav"],
    )
    assert "mixed formats" in (adapter.validate(mixed) or "")

    four_speakers = OpenAICreateSpeechRequest(
        input="\n".join(f"Speaker {idx}: text" for idx in range(4)),
        ref_audio=[f"file:///{idx}.wav" for idx in range(4)],
    )
    assert adapter.validate(four_speakers) is None

    too_many = OpenAICreateSpeechRequest(
        input="\n".join(f"Speaker {idx}: text" for idx in range(5)),
        ref_audio=[f"file:///{idx}.wav" for idx in range(5)],
    )
    assert adapter.validate(too_many) == "VibeVoice-1.5B supports at most 4 speakers per request"


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        pytest.param("speaker_embedding", [0.1, 0.2], id="speaker-embedding"),
        pytest.param("instructions", "speak softly", id="instructions"),
        pytest.param("language", "English", id="language"),
        pytest.param("ref_text", "reference transcript", id="ref-text"),
        pytest.param("ref_audio_2", "file:///second.wav", id="ref-audio-2"),
        pytest.param("task_type", "Base", id="task-type"),
        pytest.param("ambient_sound", "rain", id="ambient-sound"),
        pytest.param("duration_seconds", 1.0, id="duration-seconds"),
        pytest.param("x_vector_only_mode", True, id="x-vector-true"),
        pytest.param("x_vector_only_mode", False, id="x-vector-false"),
        pytest.param("initial_codec_chunk_frames", 4, id="initial-codec-chunk-frames"),
        pytest.param("non_streaming_mode", True, id="non-streaming-mode-true"),
        pytest.param("non_streaming_mode", False, id="non-streaming-mode-false"),
        pytest.param("word_timestamps", True, id="word-timestamps"),
    ],
)
def test_vibevoice_adapter_rejects_explicit_unsupported_fields(
    field_name: str,
    field_value: object,
) -> None:
    request_kwargs = {
        "input": "hello",
        "ref_audio": "file:///voice.wav",
        field_name: field_value,
    }
    if field_name == "speaker_embedding":
        # The shared protocol rejects a scalar ref_audio plus speaker_embedding
        # before the model adapter runs. Omit it to exercise the VibeVoice error.
        request_kwargs["ref_audio"] = None
    request = OpenAICreateSpeechRequest(**request_kwargs)

    assert f"does not support '{field_name}'" in (_adapter().validate(request) or "")


def test_vibevoice_uploaded_voice_validation_is_idempotent_and_build_resolves_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _uploaded_voice_adapter()
    request = OpenAICreateSpeechRequest(input="hello", voice="Alice")
    resolved_sources: list[str] = []

    async def resolve(source: str):
        resolved_sources.append(source)
        return np.zeros(3_200, dtype=np.float32), 24_000

    monkeypatch.setattr(adapter, "_resolve_reference", resolve)

    assert adapter.validate(request) is None
    assert adapter.validate(request) is None
    assert request.voice == "Alice"
    assert request.ref_audio is None

    prepared = asyncio.run(adapter.build(request, [], False))

    assert resolved_sources == ["data:audio/wav;base64,dGVzdA=="]
    assert request.voice is None
    assert request.ref_audio == "data:audio/wav;base64,dGVzdA=="
    assert request.ref_text is None
    assert len(prepared.prompt["multi_modal_data"]["audio"]) == 1
    # Resolution canonicalizes the mutable serving request, so later defensive
    # validation sees the same ref_audio-only request rather than a false XOR.
    assert adapter.validate(request) is None


@pytest.mark.parametrize(
    ("speech_request", "message"),
    [
        pytest.param(
            OpenAICreateSpeechRequest(
                input="hello",
                voice="alice",
                ref_audio="file:///voice.wav",
            ),
            "exactly one of 'voice' or 'ref_audio'",
            id="voice-and-inline-reference",
        ),
        pytest.param(
            OpenAICreateSpeechRequest(
                input="Speaker 0: hello\nSpeaker 1: world",
                voice="alice",
            ),
            "only supported for single-speaker",
            id="multi-speaker",
        ),
    ],
)
def test_vibevoice_uploaded_voice_rejects_conflicting_request_shapes(
    speech_request: OpenAICreateSpeechRequest,
    message: str,
) -> None:
    assert message in (_uploaded_voice_adapter().validate(speech_request) or "")


def test_vibevoice_uploaded_voice_rejects_unknown_and_embedding_profiles() -> None:
    unknown = OpenAICreateSpeechRequest(input="hello", voice="missing")
    assert "Unknown VibeVoice voice 'missing'" in (_uploaded_voice_adapter().validate(unknown) or "")

    embedding = OpenAICreateSpeechRequest(input="hello", voice="alice")
    error = _uploaded_voice_adapter(embedding_source="direct").validate(embedding) or ""
    assert "uses a speaker embedding" in error
    assert "re-upload it with an audio file" in error


def test_vibevoice_uploaded_voice_missing_audio_is_request_visible() -> None:
    adapter = _uploaded_voice_adapter(audio_data=None)
    request = OpenAICreateSpeechRequest(input="hello", voice="alice")

    assert adapter.validate(request) is None
    with pytest.raises(ValueError, match="Audio file for uploaded voice 'alice' is missing"):
        asyncio.run(adapter.build(request, [], False))


def test_vibevoice_speaker_alias_resolves_uploaded_voice() -> None:
    adapter = _uploaded_voice_adapter(voice_name="narrator")
    request = OpenAICreateSpeechRequest(input="hello", speaker="Narrator")

    assert request.voice == "Narrator"
    assert adapter.validate(request) is None


def test_vibevoice_adapter_accepts_all_supported_request_fields() -> None:
    request = OpenAICreateSpeechRequest(
        input="hello",
        model="microsoft/VibeVoice-1.5B",
        ref_audio="file:///voice.wav",
        response_format="flac",
        speed=1.25,
        stream=False,
        max_new_tokens=32,
        extra_params={
            "guidance_scale": 1.3,
            "num_diffusion_steps": 10,
            "future_extension": "ignored-for-forward-compatibility",
        },
        word_timestamps=False,
    )

    assert _adapter().validate(request) is None


@pytest.mark.parametrize(
    ("stream", "stream_format"),
    [
        (False, "audio"),
        (True, "audio"),
    ],
)
def test_vibevoice_adapter_rejects_raw_http_streaming(
    stream: bool,
    stream_format: str | None,
) -> None:
    request = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="file:///voice.wav",
        stream=stream,
        stream_format=stream_format,
        # Keep protocol validation from masking the adapter-specific rejection.
        response_format="wav",
    )
    assert request.is_raw_audio_stream()
    assert "cannot expose the terminal finish reason" in (_adapter().validate(request) or "")


@pytest.mark.parametrize(
    ("stream", "stream_format"),
    [
        (False, "audio"),
        (True, "audio"),
    ],
)
def test_vibevoice_serving_rejects_raw_streaming_before_engine_generation(
    stream: bool,
    stream_format: str | None,
) -> None:
    def generate(**_kwargs):
        raise AssertionError("VibeVoice streaming validation reached the engine")

    engine_client = SimpleNamespace(
        errored=False,
        default_sampling_params_list=[SamplingParams(max_tokens=8)],
        model_config=SimpleNamespace(async_chunk=False),
        generate=generate,
    )
    serving = object.__new__(OmniOpenAIServingSpeech)
    serving._diffusion_mode = False
    serving.engine_client = engine_client
    serving.model_config = engine_client.model_config
    serving._tts_model_type = "vibevoice"
    serving._is_tts = True
    serving._adapter = VibeVoiceTTSAdapter(SpeechServingContext(server=serving, engine_client=engine_client))
    serving._validate_ref_audio_format = lambda _: None

    async def check_model(_request):
        return None

    serving._check_model = check_model
    request = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="file:///voice.wav",
        stream=stream,
        stream_format=stream_format,
        response_format="wav",
    )

    response = asyncio.run(serving.create_speech(request))

    assert response.error.code == 400
    assert "cannot expose the terminal finish reason" in response.error.message


def test_vibevoice_sse_streams_delta_pcm_and_terminal_finish_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: list(text.encode("utf-8")))
    waveforms = [
        np.linspace(-0.2, 0.0, 3_200, dtype=np.float32),
        np.linspace(0.0, 0.2, 3_200, dtype=np.float32),
    ]

    def generate(**kwargs):
        captured.update(kwargs)

        async def outputs():
            for index, waveform in enumerate(waveforms):
                finish_reason = "length" if index == len(waveforms) - 1 else None
                yield SimpleNamespace(
                    multimodal_output={
                        "audio": torch.from_numpy(waveform),
                        "sr": torch.tensor(24_000, dtype=torch.int32),
                    },
                    request_output=SimpleNamespace(
                        outputs=[SimpleNamespace(finish_reason=finish_reason)],
                    ),
                    outputs=[SimpleNamespace(finish_reason=finish_reason)],
                    metrics={"stage_metrics": {0: {"num_tokens_out": index + 1}}},
                )

        return outputs()

    engine_client = SimpleNamespace(
        errored=False,
        default_sampling_params_list=[SamplingParams(max_tokens=2)],
        model_config=SimpleNamespace(
            async_chunk=False,
            allowed_local_media_path=None,
            allowed_media_domains=None,
        ),
        engine=SimpleNamespace(
            input_processor=SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda: tokenizer))
        ),
        generate=generate,
    )
    serving = object.__new__(OmniOpenAIServingSpeech)
    serving._diffusion_mode = False
    serving.engine_client = engine_client
    serving.model_config = engine_client.model_config
    serving._tts_model_type = "vibevoice"
    serving._is_tts = True
    serving._request_ref_audio_artifact_keys = {}
    serving._ref_audio_model_artifact_ready = set()
    serving._ref_audio_resolve_cache = {}
    serving._track_ref_audio_artifact_warmup = lambda *_args, **_kwargs: None
    serving._validate_ref_audio_format = lambda _source: None
    serving._adapter = VibeVoiceTTSAdapter(SpeechServingContext(server=serving, engine_client=engine_client))
    serving._build_speech_usage = lambda *_args, **_kwargs: SpeechTokenUsage(output_tokens=2, total_tokens=2)

    async def resolve_reference(_source: str):
        return np.zeros(3_200, dtype=np.float32), 24_000

    async def check_model(_request):
        return None

    monkeypatch.setattr(serving._adapter, "_resolve_reference", resolve_reference)
    serving._check_model = check_model
    request = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="file:///voice.wav",
        stream=True,
        response_format="pcm",
    )

    async def collect_response() -> bytes:
        response = await serving.create_speech(request)
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.encode() if isinstance(chunk, str) else chunk)
        return b"".join(chunks)

    body = asyncio.run(collect_response()).decode()
    events = [json.loads(line.removeprefix("data: ")) for line in body.splitlines() if line.startswith("data: ")]

    audio_events = [event for event in events if event["type"] == "speech.audio.delta"]
    streamed_pcm = b"".join(base64.b64decode(event["audio"]) for event in audio_events)
    expected_pcm = serving.create_audio(
        CreateAudio(
            audio_tensor=np.concatenate(waveforms),
            sample_rate=24_000,
            response_format="pcm",
            speed=1.0,
            base64_encode=False,
        )
    ).audio_data
    assert streamed_pcm == expected_pcm
    assert events[-1]["type"] == "speech.audio.done"
    assert events[-1]["finish_reason"] == "length"
    assert captured["sampling_params_list"][0].output_kind.name == "DELTA"


def test_vibevoice_serving_rejects_unsupported_field_before_engine_generation() -> None:
    def generate(**_kwargs):
        raise AssertionError("VibeVoice unsupported-field validation reached the engine")

    engine_client = SimpleNamespace(
        errored=False,
        default_sampling_params_list=[SamplingParams(max_tokens=8)],
        model_config=SimpleNamespace(async_chunk=False),
        generate=generate,
    )
    serving = object.__new__(OmniOpenAIServingSpeech)
    serving._diffusion_mode = False
    serving.engine_client = engine_client
    serving.model_config = engine_client.model_config
    serving._tts_model_type = "vibevoice"
    serving._is_tts = True
    serving._adapter = VibeVoiceTTSAdapter(SpeechServingContext(server=serving, engine_client=engine_client))
    serving._validate_ref_audio_format = lambda _: None

    async def check_model(_request):
        return None

    serving._check_model = check_model
    request = OpenAICreateSpeechRequest(
        input="hello",
        instructions="speak softly",
        ref_audio="file:///voice.wav",
    )

    response = asyncio.run(serving.create_speech(request))

    assert response.error.code == 400
    assert "does not support 'instructions'" in response.error.message


@pytest.mark.parametrize(
    ("extra_params", "message"),
    [
        ({"guidance_scale": float("nan")}, "guidance_scale must be finite"),
        ({"guidance_scale": "bad"}, "guidance_scale must be finite"),
        ({"num_diffusion_steps": 0}, "must be a positive integer"),
        ({"num_diffusion_steps": True}, "must be a positive integer"),
    ],
)
def test_vibevoice_adapter_rejects_invalid_runtime_controls(
    extra_params: dict[str, object],
    message: str,
) -> None:
    request = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="file:///voice.wav",
        extra_params=extra_params,
    )

    assert message in (_adapter().validate(request) or "")


def test_vibevoice_adapter_accepts_absent_or_valid_runtime_controls() -> None:
    adapter = _adapter()
    absent = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="file:///voice.wav",
    )
    valid = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="file:///voice.wav",
        extra_params={"guidance_scale": 1.3, "num_diffusion_steps": 10},
    )

    assert adapter.validate(absent) is None
    assert adapter.validate(valid) is None


def test_vibevoice_adapter_rejects_only_explicit_request_seed() -> None:
    adapter = _adapter()
    implicit = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="file:///voice.wav",
    )
    explicit = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="file:///voice.wav",
        seed=42,
    )

    assert adapter.validate(implicit) is None
    assert "request-level seed" in (adapter.validate(explicit) or "")


@pytest.mark.parametrize(
    ("extra_params", "seed", "message"),
    [
        ({"guidance_scale": float("inf")}, None, "guidance_scale must be finite"),
        ({"num_diffusion_steps": 0}, None, "must be a positive integer"),
        (None, 42, "request-level seed"),
    ],
)
def test_vibevoice_serving_rejects_invalid_controls_before_engine_generation(
    extra_params: dict[str, object] | None,
    seed: int | None,
    message: str,
) -> None:
    def generate(**_kwargs):
        raise AssertionError("VibeVoice invalid controls reached the engine")

    engine_client = SimpleNamespace(
        errored=False,
        default_sampling_params_list=[
            SamplingParams(
                max_tokens=8,
                seed=314,
                extra_args={
                    "guidance_scale": 1.3,
                    "num_diffusion_steps": 10,
                },
            )
        ],
        model_config=SimpleNamespace(async_chunk=False),
        generate=generate,
    )
    serving = object.__new__(OmniOpenAIServingSpeech)
    serving.engine_client = engine_client
    serving.model_config = engine_client.model_config
    serving._tts_model_type = "vibevoice"
    serving._is_tts = True
    serving._adapter = VibeVoiceTTSAdapter(SpeechServingContext(server=serving, engine_client=engine_client))
    serving._validate_ref_audio_format = lambda _: None

    async def resolve_reference(_source: str):
        return np.zeros(3_200, dtype=np.float32), 24_000

    serving._adapter._resolve_reference = resolve_reference
    request = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="file:///voice.wav",
        extra_params=extra_params,
        seed=seed,
    )

    with pytest.raises(ValueError, match=message):
        asyncio.run(serving._prepare_speech_generation(request))

    # The valid deploy seed and controls remain internal defaults. Only
    # explicitly invalid request fields are rejected before generate().
    assert engine_client.default_sampling_params_list[0].seed == 314
    assert engine_client.default_sampling_params_list[0].extra_args == {
        "guidance_scale": 1.3,
        "num_diffusion_steps": 10,
    }


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
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: list(text.encode("utf-8")))

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
            input_processor=SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda: tokenizer))
        ),
        generate=generate,
    )
    serving = object.__new__(OmniOpenAIServingSpeech)
    serving.engine_client = engine_client
    serving.model_config = engine_client.model_config
    serving._tts_model_type = "vibevoice"
    serving._is_tts = True
    serving._adapter = VibeVoiceTTSAdapter(SpeechServingContext(server=serving, engine_client=engine_client))
    serving._track_ref_audio_artifact_warmup = lambda *args, **kwargs: None
    serving._validate_ref_audio_format = lambda _: None

    async def resolve(_source: str):
        return np.zeros(3200, dtype=np.float32), 24_000

    monkeypatch.setattr(serving._adapter, "_resolve_reference", resolve)
    request = OpenAICreateSpeechRequest(input="hello", ref_audio="ref-a")

    request_id, generator, _ = asyncio.run(serving._prepare_speech_generation(request, request_id="speech-fixed"))

    assert request_id == "speech-fixed"
    assert generator == "generator"
    assert captured["prompt"]["multi_modal_uuids"] == {"audio": ["speech-fixed:audio:0"]}
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


@pytest.mark.parametrize(
    ("model_type", "finish_reason", "expected_header"),
    [
        pytest.param("vibevoice", "length", "length", id="vibevoice-length"),
        pytest.param("vibevoice", "stop", "stop", id="vibevoice-stop"),
        pytest.param("other_tts", "length", None, id="non-vibevoice"),
    ],
)
def test_nonstreaming_speech_response_exposes_vibevoice_finish_reason(
    model_type: str,
    finish_reason: str,
    expected_header: str | None,
) -> None:
    serving = object.__new__(OmniOpenAIServingSpeech)
    serving._diffusion_mode = False
    serving._tts_model_type = model_type
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
        request_output=SimpleNamespace(
            outputs=[SimpleNamespace(finish_reason=finish_reason)],
        ),
        outputs=[],
    )

    async def generator():
        yield final

    async def prepare(*_args, **_kwargs):
        return "vibevoice-finish-reason", generator(), {}

    async def check_model(_request):
        return None

    serving._prepare_speech_generation = prepare
    serving._check_model = check_model
    request = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="ref",
        response_format="wav",
    )

    response = asyncio.run(serving.create_speech(request))

    assert response.status_code == 200
    assert response.headers.get("X-Finish-Reason") == expected_header


def test_vibevoice_batch_returns_mixed_item_results_and_finish_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_prompts: list[str] = []
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: list(text.encode("utf-8")))

    def generate(**kwargs):
        prompt = kwargs["prompt"]["prompt"]
        generated_prompts.append(prompt)
        finish_reason = "length" if "force length" in prompt else "stop"
        waveform = torch.from_numpy(np.linspace(-0.25, 0.25, 3_200, dtype=np.float32))
        multimodal_output = {
            "audio": waveform,
            "sr": torch.tensor(24_000, dtype=torch.int32),
        }
        final = SimpleNamespace(
            multimodal_output=multimodal_output,
            request_output=SimpleNamespace(
                outputs=[SimpleNamespace(finish_reason=finish_reason)],
            ),
            outputs=[],
            metrics={"stage_metrics": {0: {"num_tokens_out": 1}}},
        )

        async def outputs():
            yield final

        return outputs()

    engine_client = SimpleNamespace(
        errored=False,
        default_sampling_params_list=[SamplingParams(max_tokens=32)],
        model_config=SimpleNamespace(
            async_chunk=False,
            allowed_local_media_path=None,
            allowed_media_domains=None,
        ),
        engine=SimpleNamespace(
            input_processor=SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda: tokenizer))
        ),
        generate=generate,
    )
    serving = object.__new__(OmniOpenAIServingSpeech)
    serving._diffusion_mode = False
    serving._batch_max_items = 32
    serving.engine_client = engine_client
    serving.model_config = engine_client.model_config
    serving._tts_model_type = "vibevoice"
    serving._is_tts = True
    serving._request_ref_audio_artifact_keys = {}
    serving._ref_audio_model_artifact_ready = set()
    serving._ref_audio_resolve_cache = {}
    serving._track_ref_audio_artifact_warmup = lambda *_args, **_kwargs: None
    serving._validate_ref_audio_format = lambda _source: None
    serving.uploaded_speakers = {
        "alice": {
            "name": "alice",
            "embedding_source": "audio",
            "ref_text": "stored transcript",
        }
    }
    serving._get_uploaded_audio_data = lambda _voice: "data:audio/wav;base64,dGVzdA=="
    serving._adapter = VibeVoiceTTSAdapter(SpeechServingContext(server=serving, engine_client=engine_client))
    serving._build_speech_usage = lambda *_args, **_kwargs: SpeechTokenUsage(output_tokens=1, total_tokens=1)

    async def resolve_reference(_source: str):
        return np.zeros(3_200, dtype=np.float32), 24_000

    async def check_model(_request):
        return None

    monkeypatch.setattr(serving._adapter, "_resolve_reference", resolve_reference)
    serving._check_model = check_model
    batch = BatchSpeechRequest(
        items=[
            SpeechBatchItem(input="force length", voice="alice", max_new_tokens=2),
            SpeechBatchItem(
                input="invalid item",
                ref_audio="file:///voice.wav",
                instructions="unsupported",
            ),
            SpeechBatchItem(input="natural stop", ref_audio="file:///voice.wav"),
        ]
    )

    response = asyncio.run(serving.create_speech_batch(batch))

    assert [item.status for item in response.results] == ["success", "error", "success"]
    assert response.results[0].finish_reason == "length"
    assert response.results[0].audio_data is not None
    assert "does not support 'instructions'" in (response.results[1].error or "")
    assert response.results[1].finish_reason is None
    assert response.results[2].finish_reason == "stop"
    assert response.succeeded == 2
    assert response.failed == 1
    assert len(generated_prompts) == 2


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
