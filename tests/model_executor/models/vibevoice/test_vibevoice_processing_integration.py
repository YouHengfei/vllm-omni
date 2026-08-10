# SPDX-License-Identifier: Apache-2.0
"""Processing-only integration coverage against the Microsoft processor.

These tests deliberately stop at stage-0 input processing. They do not enable
VibeVoice stateful preprocess/postprocess or waveform inference.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, PretrainedConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.multimodal.processing import InputProcessingContext

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    PreparedRequest,
    SpeechServingContext,
)
from vllm_omni.entrypoints.openai.tts_adapters.vibevoice import VibeVoiceTTSAdapter
from vllm_omni.model_executor.models.vibevoice.processing_vibevoice import (
    SAMPLE_RATE,
    VibeVoiceDummyInputsBuilder,
    VibeVoiceMultiModalProcessor,
    VibeVoiceProcessingInfo,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _fixture_paths() -> tuple[Path, Path]:
    model_root = os.getenv("VIBEVOICE_TEST_MODEL_ROOT")
    official_repo = os.getenv("VIBEVOICE_OFFICIAL_REPO")
    if not model_root:
        pytest.skip("Set VIBEVOICE_TEST_MODEL_ROOT for real-tokenizer processing parity")
    if not official_repo:
        pytest.skip("Set VIBEVOICE_OFFICIAL_REPO for Microsoft processor parity")

    tokenizer_path = Path(model_root) / "VibeVoice-1.5B-hf"
    repo_path = Path(official_repo)
    if not (tokenizer_path / "tokenizer.json").is_file():
        pytest.skip(f"VibeVoice tokenizer fixture not found at {tokenizer_path}")
    if not (repo_path / "vibevoice/processor/vibevoice_processor.py").is_file():
        pytest.skip(f"Microsoft VibeVoice processor not found at {repo_path}")
    return tokenizer_path, repo_path


def _load_microsoft_processor_class(repo_path: Path):
    """Import only the official processor package, avoiding repo model registration.

    Importing Microsoft ``vibevoice.__init__`` also registers model classes and
    collides with the pinned Transformers VibeVoice acoustic registrations. A
    namespace package gives this parity test the processor source without
    importing the unrelated model package initializer.
    """
    package_root = repo_path / "vibevoice"
    processor_root = package_root / "processor"

    vibevoice_pkg = types.ModuleType("_vv_official")
    vibevoice_pkg.__path__ = [str(package_root)]
    processor_pkg = types.ModuleType("_vv_official.processor")
    processor_pkg.__path__ = [str(processor_root)]
    sys.modules.setdefault("_vv_official", vibevoice_pkg)
    sys.modules.setdefault("_vv_official.processor", processor_pkg)
    module = importlib.import_module("_vv_official.processor.vibevoice_processor")
    return module.VibeVoiceProcessor


def _real_components():
    tokenizer_path, repo_path = _fixture_paths()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)

    # Microsoft Processor names these IDs speech_*, while the HF PR and Omni
    # runtime expose the same checkpoint IDs as audio_* / vision_* tokens.
    tokenizer.speech_start_id = 151652
    tokenizer.speech_end_id = 151653
    tokenizer.speech_diffusion_id = 151654

    config = PretrainedConfig()
    config.audio_bos_token_id = 151652
    config.audio_eos_token_id = 151653
    config.audio_token_id = 151654
    multimodal_config = MultiModalConfig(limit_per_prompt={"audio": 8})
    model_config = SimpleNamespace(
        model=str(tokenizer_path),
        hf_config=config,
        multimodal_config=multimodal_config,
        dtype=torch.float32,
        encoder_config=None,
        max_model_len=65_536,
        get_multimodal_config=lambda: multimodal_config,
    )
    context = InputProcessingContext(model_config, tokenizer=tokenizer)
    info = VibeVoiceProcessingInfo(context)
    processor = VibeVoiceMultiModalProcessor(
        info,
        VibeVoiceDummyInputsBuilder(info),
        cache=None,
    )

    engine_client = SimpleNamespace(
        engine=SimpleNamespace(
            input_processor=SimpleNamespace(
                renderer=SimpleNamespace(get_tokenizer=lambda: tokenizer)
            )
        )
    )
    adapter = VibeVoiceTTSAdapter(
        SpeechServingContext(
            server=SimpleNamespace(
                model_config=SimpleNamespace(
                    allowed_local_media_path=None,
                    allowed_media_domains=None,
                ),
                _validate_ref_audio_format=lambda _: None,
            ),
            engine_client=engine_client,
        )
    )

    official_cls = _load_microsoft_processor_class(repo_path)
    official = official_cls(
        tokenizer=tokenizer,
        audio_processor=None,
        speech_tok_compress_ratio=3200,
        # Token parity is independent of waveform normalization.
        db_normalize=False,
    )
    return tokenizer, adapter, info, processor, official


def _build_adapter_request(
    adapter: VibeVoiceTTSAdapter,
    request: OpenAICreateSpeechRequest,
    waveforms: list[np.ndarray],
    request_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> PreparedRequest:
    references = list(request.ref_audio) if isinstance(request.ref_audio, list) else [request.ref_audio]
    by_source = dict(zip(references, waveforms, strict=True))

    async def resolve(source: str):
        return by_source[source], SAMPLE_RATE

    monkeypatch.setattr(adapter, "_resolve_reference", resolve)
    prepared = asyncio.run(adapter.build(request, [], True))
    return adapter.finalize_prepared_request(prepared, request_id)


def test_adapter_and_mm_processor_token_ids_match_microsoft_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, adapter, info, processor, official = _real_components()
    script = "Speaker 1: first line\nSpeaker 2: second line\nSpeaker 1: third line"
    waveforms = [
        np.linspace(-0.2, 0.2, 3_200, dtype=np.float32),
        np.linspace(0.2, -0.2, 6_401, dtype=np.float32),
    ]
    request = OpenAICreateSpeechRequest(
        input=script,
        ref_audio=["ref-a", "ref-b"],
    )
    prepared = _build_adapter_request(
        adapter,
        request,
        waveforms,
        "processing-parity-request",
        monkeypatch,
    )

    mm_items = info.parse_mm_data(
        {"audio": [(waveform, SAMPLE_RATE) for waveform in waveforms]}
    )
    omni = processor(
        prepared.prompt["prompt_token_ids"],
        mm_items=mm_items,
        mm_uuid_items=prepared.prompt["multi_modal_uuids"],
    )
    microsoft = official(
        text=script,
        voice_samples=waveforms,
        padding=False,
    )

    assert omni["prompt_token_ids"] == microsoft["input_ids"][0]
    assert [item.length for item in omni["mm_placeholders"]["audio"]] == [1, 3]
    assert prepared.prompt["multi_modal_uuids"] == {
        "audio": [
            "processing-parity-request:audio:0",
            "processing-parity-request:audio:1",
        ]
    }


def test_adapter_token_prompt_uuid_path_is_stable_through_mm_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, adapter, info, processor, _ = _real_components()
    waveform = np.zeros(3_200, dtype=np.float32)
    request = OpenAICreateSpeechRequest(input="hello", ref_audio="ref-a")

    first = _build_adapter_request(
        adapter,
        request,
        [waveform],
        "stable-request",
        monkeypatch,
    )
    mm_items = info.parse_mm_data({"audio": [(waveform, SAMPLE_RATE)]})
    first_processed = processor(
        first.prompt["prompt_token_ids"],
        mm_items=mm_items,
        mm_uuid_items=first.prompt["multi_modal_uuids"],
    )
    repeated = processor(
        first.prompt["prompt_token_ids"],
        mm_items=mm_items,
        mm_uuid_items=first.prompt["multi_modal_uuids"],
    )

    second = adapter.finalize_prepared_request(
        PreparedRequest(
            prompt={
                "prompt": first.prompt["prompt"],
                "multi_modal_data": first.prompt["multi_modal_data"],
            },
            model_type="vibevoice",
        ),
        "different-request",
    )
    second_processed = processor(
        second.prompt["prompt_token_ids"],
        mm_items=mm_items,
        mm_uuid_items=second.prompt["multi_modal_uuids"],
    )

    assert first_processed["prompt_token_ids"] == first.prompt["prompt_token_ids"]
    assert first_processed["mm_hashes"] == repeated["mm_hashes"]
    assert first_processed["mm_hashes"] != second_processed["mm_hashes"]
