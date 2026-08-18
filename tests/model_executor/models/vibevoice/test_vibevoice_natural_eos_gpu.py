# SPDX-License-Identifier: Apache-2.0
"""Natural-EOS generation through the real TP=2 AsyncOmni runtime."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import traceback
from pathlib import Path
from queue import Empty
from typing import Any

import pytest
import torch

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.accelerator.is_available() or torch.accelerator.device_count() < 2,
        reason="Two CUDA devices are required",
    ),
]

_WORKER_EXTENSION = "tests.helpers.vibevoice_worker_extension.VibeVoiceWorkerExtensionForTest"
_NATURAL_SEED = 12_345
_AUDIO_BOS = 151652
_AUDIO_EOS = 151653
_AUDIO_TOKEN = 151654
_EOS = 151643


def _run_natural_eos_generation(
    model_path: str,
    tokenizer_path: str,
    deploy_path: str,
    queue: Any,
) -> None:
    try:
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        import numpy as np
        import soundfile as sf
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        from vllm_omni.entrypoints.async_omni import AsyncOmni
        from vllm_omni.entrypoints.openai.protocol.audio import (
            OpenAICreateSpeechRequest,
        )
        from vllm_omni.entrypoints.openai.tts_adapters.base import (
            SpeechServingContext,
        )
        from vllm_omni.entrypoints.openai.tts_adapters.vibevoice import (
            VibeVoiceTTSAdapter,
        )

        async def run() -> dict[str, Any]:
            engine = AsyncOmni(
                model=model_path,
                tokenizer=tokenizer_path,
                deploy_config=deploy_path,
                stage_overrides={"0": {"worker_extension_cls": _WORKER_EXTENSION}},
                stage_init_timeout=300,
                init_timeout=600,
                log_stats=False,
            )
            try:
                armed = await engine.collective_rpc(
                    method="vibevoice_test_arm_natural_lifecycle_trace",
                    args=(_NATURAL_SEED,),
                    timeout=60,
                    stage_ids=[0],
                )
                assert len(armed) == 1
                assert [item["rank"] for item in armed[0]] == [0, 1]
                assert all(item["armed"] for item in armed[0])

                server = type("Server", (), {})()
                server.model_config = type(
                    "ModelConfig",
                    (),
                    {
                        "allowed_local_media_path": None,
                        "allowed_media_domains": None,
                    },
                )()
                server._validate_ref_audio_format = lambda _source: None
                adapter = VibeVoiceTTSAdapter(SpeechServingContext(server=server, engine_client=engine.engine))

                asset_root = Path(__file__).parents[3] / "assets"
                four_speaker_sources = {
                    f"test-only-four-speaker-{index}": asset_root / relative
                    for index, relative in enumerate(
                        (
                            "cosyvoice3/zero_shot_prompt.wav",
                            "glm_tts/jiayan_zh.wav",
                            "indextts2/ref_audio.wav",
                            "qwen3_tts/clone_2.wav",
                        )
                    )
                }

                async def resolve_reference(source: str):
                    path = four_speaker_sources.get(source)
                    if path is not None:
                        waveform, sample_rate = sf.read(
                            path,
                            dtype="float32",
                            always_2d=False,
                        )
                        return np.asarray(waveform, dtype=np.float32), int(sample_rate)
                    return (
                        np.linspace(-0.2, 0.2, 3_200, dtype=np.float32),
                        24_000,
                    )

                adapter._resolve_reference = resolve_reference

                async def generate_request(
                    request_id: str,
                    request: OpenAICreateSpeechRequest,
                    *,
                    max_tokens: int,
                ) -> dict[str, Any]:
                    prepared = await adapter.build(request, [], True)
                    prepared = adapter.finalize_prepared_request(
                        prepared,
                        request_id,
                    )
                    params = SamplingParams(
                        max_tokens=max_tokens,
                        temperature=0.0,
                        allowed_token_ids=[151652, _AUDIO_EOS, _AUDIO_TOKEN, _EOS],
                        stop_token_ids=[_EOS],
                        detokenize=False,
                        output_kind=RequestOutputKind.FINAL_ONLY,
                        # Keep production controls: deploy defaults are 1.3/10.
                        extra_args={
                            "guidance_scale": 1.3,
                            "num_diffusion_steps": 10,
                        },
                    )
                    outputs = []
                    async for output in engine.generate(
                        prompt=prepared.prompt,
                        request_id=request_id,
                        sampling_params_list=[params],
                        output_modalities=["audio"],
                    ):
                        outputs.append(output)
                    final = outputs[-1]
                    completion = final.outputs[0]
                    token_ids = list(completion.token_ids)
                    audio = final.multimodal_output.get("audio")
                    sample_rate = final.multimodal_output.get("sr")
                    return {
                        "request_id": request_id,
                        "token_ids": token_ids,
                        "finish_reason": completion.finish_reason,
                        "finished": bool(final.finished),
                        "audio_shape": (tuple(audio.shape) if isinstance(audio, torch.Tensor) else None),
                        "audio_dtype": (str(audio.dtype) if isinstance(audio, torch.Tensor) else None),
                        "audio_device": (audio.device.type if isinstance(audio, torch.Tensor) else None),
                        "audio_finite": bool(isinstance(audio, torch.Tensor) and torch.isfinite(audio).all()),
                        "sample_rate": int(
                            sample_rate.item() if isinstance(sample_rate, torch.Tensor) else sample_rate
                        ),
                    }

                short = await generate_request(
                    "vibevoice-natural-eos-short",
                    OpenAICreateSpeechRequest(
                        input="Hello.",
                        ref_audio="test-only-reference-short",
                    ),
                    max_tokens=256,
                )
                dialogue = await generate_request(
                    "vibevoice-natural-eos-dialogue",
                    OpenAICreateSpeechRequest(
                        input=(
                            "Speaker 4: Hello, this is the first sentence.\n"
                            "Speaker 9: Welcome. This is a longer second sentence "
                            "for natural multi-speaker speech."
                        ),
                        ref_audio=[
                            "test-only-reference-speaker-0",
                            "test-only-reference-speaker-1",
                        ],
                    ),
                    max_tokens=512,
                )
                four_speaker = await generate_request(
                    "vibevoice-natural-eos-four-speaker",
                    OpenAICreateSpeechRequest(
                        input=(
                            "Speaker 0: Welcome.\n"
                            "Speaker 1: It is good to be here.\n"
                            "Speaker 2: Let us begin.\n"
                            "Speaker 3: Thank you."
                        ),
                        ref_audio=list(four_speaker_sources),
                    ),
                    max_tokens=1_024,
                )

                lifecycle = await engine.collective_rpc(
                    method="vibevoice_test_take_natural_lifecycle_trace",
                    timeout=60,
                    stage_ids=[0],
                )
                runtime_state = await engine.collective_rpc(
                    method="vibevoice_test_runtime_state",
                    timeout=60,
                    stage_ids=[0],
                )
                return {
                    "short": short,
                    "dialogue": dialogue,
                    "four_speaker": four_speaker,
                    "lifecycle": lifecycle[0],
                    "runtime_state": runtime_state[0],
                }
            finally:
                engine.shutdown()

        queue.put(asyncio.run(run()))
    except Exception:
        queue.put({"error": traceback.format_exc()})


@pytest.fixture(scope="module")
def natural_eos_result() -> dict[str, Any]:
    model_root = os.getenv("VIBEVOICE_TEST_MODEL_ROOT")
    if not model_root:
        pytest.skip("Set VIBEVOICE_TEST_MODEL_ROOT to run natural-EOS generation")

    model_path = Path(model_root) / "VibeVoice"
    tokenizer_path = Path(model_root) / "VibeVoice-1.5B-hf"
    deploy_path = Path(__file__).parents[4] / "vllm_omni/deploy/vibevoice.yaml"
    if not (model_path / "model.safetensors.index.json").is_file():
        pytest.fail(f"Official VibeVoice checkpoint not found at {model_path}")
    if not (tokenizer_path / "tokenizer.json").is_file():
        pytest.skip(f"VibeVoice tokenizer fixture not found at {tokenizer_path}")

    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_run_natural_eos_generation,
        args=(
            str(model_path),
            str(tokenizer_path),
            str(deploy_path),
            queue,
        ),
    )
    process.start()
    process.join(timeout=900)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("VibeVoice natural-EOS subprocess timed out")
    try:
        result = queue.get(timeout=10)
    except Empty:
        pytest.fail(f"VibeVoice natural-EOS subprocess exited without a result: exitcode={process.exitcode}")
    assert "error" not in result, result.get("error")
    assert process.exitcode == 0
    return result


@pytest.mark.parametrize(
    ("case", "expected_segments"),
    [("short", 1), ("dialogue", 2), ("four_speaker", None)],
)
def test_natural_generation_reaches_audio_eos_then_model_eos(
    natural_eos_result: dict[str, Any],
    case: str,
    expected_segments: int | None,
) -> None:
    result = natural_eos_result[case]
    token_ids = result["token_ids"]
    assert result["finished"] is True
    assert result["finish_reason"] == "stop"
    assert token_ids[-2:] == [_AUDIO_EOS, _EOS]
    segment_count = token_ids.count(_AUDIO_EOS)
    assert segment_count >= 1
    if expected_segments is not None:
        # Output segment count is model-selected (see comment below); the
        # expected value is a lower bound (>= one segment per speaker turn),
        # not an exact count — the conv-cache reset at audio_bos changes the
        # AR token path and may yield more segments than the leaky baseline.
        assert segment_count >= expected_segments
    # Output audio segments are model-selected and are not specified to map
    # one-to-one to input speaker turns. Every generated segment after the
    # prompt's initial BOS must still have a generated BOS, and all segments
    # must close before model EOS.
    assert token_ids.count(_AUDIO_BOS) == segment_count - 1
    assert token_ids.count(_EOS) == 1
    assert token_ids.count(_AUDIO_TOKEN) >= segment_count
    assert set(token_ids) <= {_AUDIO_BOS, _AUDIO_EOS, _AUDIO_TOKEN, _EOS}


@pytest.mark.parametrize("case", ["short", "dialogue", "four_speaker"])
def test_natural_generation_publishes_every_audio_transition(
    natural_eos_result: dict[str, Any],
    case: str,
) -> None:
    result = natural_eos_result[case]
    token_ids = result["token_ids"]
    assert result["audio_device"] == "cpu"
    assert result["audio_dtype"] == "torch.float32"
    assert result["audio_shape"] == (token_ids.count(_AUDIO_TOKEN) * 3_200,)
    assert result["audio_finite"] is True
    assert result["sample_rate"] == 24_000


def test_natural_audio_eos_frees_segment_negative_branch_on_both_tp_ranks(
    natural_eos_result: dict[str, Any],
) -> None:
    lifecycle = natural_eos_result["lifecycle"]
    assert [item["rank"] for item in lifecycle] == [0, 1]
    for rank_payload in lifecycle:
        events = rank_payload["events"]
        for case, result_key in (
            ("short", "short"),
            ("dialogue", "dialogue"),
            ("four-speaker", "four_speaker"),
        ):
            expected_segments = natural_eos_result[result_key]["token_ids"].count(_AUDIO_EOS)
            request_events = [event for event in events if f"natural-eos-{case}" in event["request_id"]]
            initial_starts = [event for event in request_events if event["event"] == "start_audio_segment"]
            generated_starts = [
                event
                for event in request_events
                if event["event"] == "process_sampled_token" and event["token_id"] == _AUDIO_BOS
            ]
            audio_eos_events = [
                event
                for event in request_events
                if event["event"] == "process_sampled_token" and event["token_id"] == _AUDIO_EOS
            ]
            assert len(initial_starts) == 1
            assert len(generated_starts) == expected_segments - 1
            assert len(audio_eos_events) == expected_segments
            assert initial_starts[0]["in_audio_segment"] is True
            assert initial_starts[0]["negative_tokens"] == 0
            for generated_start in generated_starts:
                assert generated_start["in_audio_segment"] is True
                assert generated_start["negative_tokens_before"] == 0
                assert generated_start["negative_tokens_after"] == 0
            for terminal in audio_eos_events:
                assert terminal["in_audio_segment"] is False
                assert terminal["audio_token_count"] > 0
                assert terminal["negative_tokens_before"] > 0
                assert terminal["negative_tokens_after"] == 0

    runtime_state = natural_eos_result["runtime_state"]
    assert [item["rank"] for item in runtime_state] == [0, 1]
    for rank_state in runtime_state:
        branch = rank_state["named_branches"]["negative"]
        assert branch["requests"] == {}
        assert branch["num_free_blocks"] == branch["num_blocks"]
