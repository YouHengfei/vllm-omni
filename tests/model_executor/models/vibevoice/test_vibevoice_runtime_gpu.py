# SPDX-License-Identifier: Apache-2.0
"""Real AsyncOmni runtime tests for VibeVoice.

Merged from test_vibevoice_omni_ar_runtime_gpu.py,
test_vibevoice_natural_eos_gpu.py, test_vibevoice_async_omni_cleanup_gpu.py,
and test_vibevoice_engine_core_gpu.py.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import tempfile
import time
import traceback
from pathlib import Path
from queue import Empty
from typing import Any

import pytest
import torch

_WORKER_EXTENSION = "tests.helpers.vibevoice_worker_extension.VibeVoiceWorkerExtensionForTest"

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _omni_ar_worker(
    model_path: str,
    tokenizer_path: str,
    deploy_path: str,
    queue: Any,
) -> None:
    try:
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        import numpy as np
        from vllm import SamplingParams

        from vllm_omni.entrypoints.async_omni import AsyncOmni
        from vllm_omni.model_executor.models.vibevoice.pipeline import (
            VIBEVOICE_VALID_TOKEN_IDS,
        )

        async def run() -> dict[str, Any]:
            engine = AsyncOmni(
                model=model_path,
                tokenizer=tokenizer_path,
                stage_configs_path=deploy_path,
                stage_overrides={"0": {"worker_extension_cls": _WORKER_EXTENSION}},
                stage_init_timeout=300,
                init_timeout=600,
                log_stats=False,
            )
            try:
                stage = engine.stage_configs[0]
                scheduler_cls = str(stage.engine_args.scheduler_cls)
                worker_type = str(stage.engine_args.worker_type)

                tokenizer = engine.engine.input_processor.renderer.get_tokenizer()
                prompt_text = "<|vision_start|><|vision_pad|><|vision_end|> Speech output:\n<|vision_start|>"

                def make_prompt(request_id: str) -> dict[str, Any]:
                    return {
                        "prompt": prompt_text,
                        "prompt_token_ids": tokenizer.encode(
                            prompt_text,
                            add_special_tokens=False,
                        ),
                        "multi_modal_data": {"audio": [(np.zeros(3_200, dtype=np.float32), 24_000)]},
                        "multi_modal_uuids": {"audio": [f"{request_id}:audio:0"]},
                    }

                prompt = make_prompt("omni-ar-processing-smoke")
                params = SamplingParams(
                    max_tokens=1,
                    temperature=0.0,
                    allowed_token_ids=list(VIBEVOICE_VALID_TOKEN_IDS),
                    stop_token_ids=[151643],
                    detokenize=False,
                )

                outputs = []
                async for output in engine.generate(
                    prompt=prompt,
                    request_id="omni-ar-processing-smoke",
                    sampling_params_list=[params],
                ):
                    outputs.append(output)

                final = outputs[-1]
                token_ids = list(final.outputs[0].token_ids)
                mm_output = getattr(final, "multimodal_output", None)
                tensors = getattr(mm_output, "tensors", {}) if mm_output is not None else {}
                audio = tensors.get("audio") if isinstance(tensors, dict) else None

                # Force three real audio-token transitions. The final sampled
                # token reaches max_tokens and is decoded by the capability-
                # gated post-sample drain without another Qwen step.
                forced_params = SamplingParams(
                    max_tokens=3,
                    temperature=0.0,
                    allowed_token_ids=[151654],
                    stop_token_ids=[151643],
                    detokenize=False,
                    extra_args={
                        "guidance_scale": 1.3,
                        "num_diffusion_steps": 1,
                    },
                )
                forced_outputs = []
                async for output in engine.generate(
                    prompt=make_prompt("omni-ar-negative-step-smoke"),
                    request_id="omni-ar-negative-step-smoke",
                    sampling_params_list=[forced_params],
                ):
                    forced_outputs.append(output)
                forced_final = forced_outputs[-1]
                forced_token_ids = list(forced_final.outputs[0].token_ids)
                forced_mm_output = getattr(
                    forced_final.outputs[0],
                    "multimodal_output",
                    None,
                )
                if forced_mm_output is None:
                    forced_mm_output = getattr(
                        forced_final,
                        "multimodal_output",
                        None,
                    )
                forced_audio = forced_mm_output.get("audio") if forced_mm_output is not None else None
                forced_sample_rate = forced_mm_output.get("sr") if forced_mm_output is not None else None
                forced_runtime_state = await engine.collective_rpc(
                    method="vibevoice_test_runtime_state",
                    timeout=60,
                    stage_ids=[0],
                )

                armed = await engine.collective_rpc(
                    method="vibevoice_test_arm_concurrency_trace",
                    timeout=60,
                    stage_ids=[0],
                )
                assert all(item["armed"] for item in armed[0])

                concurrent_params = SamplingParams(
                    max_tokens=8,
                    temperature=0.0,
                    allowed_token_ids=[151654],
                    stop_token_ids=[151643],
                    detokenize=False,
                    extra_args={
                        "guidance_scale": 1.3,
                        "num_diffusion_steps": 1,
                    },
                )

                async def generate_concurrent(request_id: str) -> dict[str, Any]:
                    outputs = []
                    request_params = concurrent_params.clone()
                    # Async scheduling pipelines A two sampled-token iterations
                    # ahead while B finishes prefill. Offset its cap so both
                    # terminal sampled tokens are drained in one runner batch.
                    request_params.max_tokens = 10 if request_id.endswith("-a") else 8
                    async for output in engine.generate(
                        prompt=make_prompt(request_id),
                        request_id=request_id,
                        sampling_params_list=[request_params],
                        output_modalities=["audio"],
                    ):
                        outputs.append(output)
                    final_output = outputs[-1]
                    completion = final_output.outputs[0]
                    mm = getattr(completion, "multimodal_output", None)
                    if mm is None:
                        mm = final_output.multimodal_output
                    request_audio = mm.get("audio") if mm is not None else None
                    return {
                        "token_ids": list(completion.token_ids),
                        "finish_reason": completion.finish_reason,
                        "audio_shape": (
                            tuple(request_audio.shape) if isinstance(request_audio, torch.Tensor) else None
                        ),
                        "audio_finite": bool(
                            isinstance(request_audio, torch.Tensor) and torch.isfinite(request_audio).all()
                        ),
                    }

                concurrent_results = await asyncio.gather(
                    generate_concurrent("omni-ar-concurrent-a"),
                    generate_concurrent("omni-ar-concurrent-b"),
                )
                concurrency_trace = await engine.collective_rpc(
                    method="vibevoice_test_take_concurrency_trace",
                    timeout=60,
                    stage_ids=[0],
                )
                concurrent_runtime_state = await engine.collective_rpc(
                    method="vibevoice_test_runtime_state",
                    timeout=60,
                    stage_ids=[0],
                )

                abort_armed = await engine.collective_rpc(
                    method="vibevoice_test_arm_concurrency_trace",
                    timeout=60,
                    stage_ids=[0],
                )
                assert all(item["armed"] for item in abort_armed[0])

                async def generate_abort_case(
                    request_id: str,
                    max_tokens: int,
                ) -> dict[str, Any]:
                    params = concurrent_params.clone()
                    params.max_tokens = max_tokens
                    outputs = []
                    async for output in engine.generate(
                        prompt=make_prompt(request_id),
                        request_id=request_id,
                        sampling_params_list=[params],
                        output_modalities=["audio"],
                    ):
                        outputs.append(output)
                    final_output = outputs[-1]
                    completion = final_output.outputs[0]
                    mm = getattr(completion, "multimodal_output", None)
                    if mm is None:
                        mm = final_output.multimodal_output
                    request_audio = mm.get("audio") if mm is not None else None
                    return {
                        "token_ids": list(completion.token_ids),
                        "finish_reason": completion.finish_reason,
                        "audio_shape": (
                            tuple(request_audio.shape) if isinstance(request_audio, torch.Tensor) else None
                        ),
                    }

                abort_task = asyncio.create_task(
                    generate_abort_case("omni-ar-abort-a", 32),
                )
                survivor_task = asyncio.create_task(
                    generate_abort_case("omni-ar-abort-survivor", 8),
                )
                observed_two_resident = False
                for _ in range(50):
                    resident_state = await engine.collective_rpc(
                        method="vibevoice_test_runtime_state",
                        timeout=60,
                        stage_ids=[0],
                    )
                    if all(
                        len(rank_state["named_branches"]["negative"]["requests"]) == 2
                        for rank_state in resident_state[0]
                    ):
                        observed_two_resident = True
                        break
                    await asyncio.sleep(0.02)
                assert observed_two_resident

                abort_task.cancel()
                await engine.abort("omni-ar-abort-a")
                abort_cancelled = False
                try:
                    await abort_task
                except asyncio.CancelledError:
                    abort_cancelled = True
                survivor_result = await survivor_task
                abort_trace = await engine.collective_rpc(
                    method="vibevoice_test_take_concurrency_trace",
                    timeout=60,
                    stage_ids=[0],
                )
                abort_runtime_state = await engine.collective_rpc(
                    method="vibevoice_test_runtime_state",
                    timeout=60,
                    stage_ids=[0],
                )

                return {
                    "scheduler_cls": scheduler_cls,
                    "worker_type": worker_type,
                    "num_outputs": len(outputs),
                    "finished": bool(final.finished),
                    "token_ids": token_ids,
                    "audio_shape": tuple(audio.shape) if isinstance(audio, torch.Tensor) else None,
                    "forced_audio_token_ids": forced_token_ids,
                    "forced_finished": bool(forced_final.finished),
                    "forced_audio_shape": (
                        tuple(forced_audio.shape) if isinstance(forced_audio, torch.Tensor) else None
                    ),
                    "forced_audio_dtype": (str(forced_audio.dtype) if isinstance(forced_audio, torch.Tensor) else None),
                    "forced_audio_finite": (
                        bool(torch.isfinite(forced_audio).all()) if isinstance(forced_audio, torch.Tensor) else False
                    ),
                    "forced_sample_rate": (
                        int(forced_sample_rate.item())
                        if isinstance(forced_sample_rate, torch.Tensor)
                        else forced_sample_rate
                    ),
                    "forced_runtime_state": forced_runtime_state[0],
                    "concurrent_results": concurrent_results,
                    "concurrency_trace": concurrency_trace[0],
                    "concurrent_runtime_state": concurrent_runtime_state[0],
                    "abort_cancelled": abort_cancelled,
                    "abort_survivor_result": survivor_result,
                    "abort_trace": abort_trace[0],
                    "abort_runtime_state": abort_runtime_state[0],
                }
            finally:
                engine.shutdown()

        queue.put(asyncio.run(run()))
    except Exception:
        queue.put({"error": traceback.format_exc()})


@pytest.fixture(scope="module")
def omni_ar_processing_result() -> dict[str, Any]:
    model_root = os.getenv("VIBEVOICE_TEST_MODEL_ROOT")
    if not model_root:
        pytest.skip("Set VIBEVOICE_TEST_MODEL_ROOT to run the Omni AR runtime smoke")

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
        target=_omni_ar_worker,
        args=(
            str(model_path),
            str(tokenizer_path),
            str(deploy_path),
            queue,
        ),
    )
    process.start()
    process.join(timeout=420)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("VibeVoice Omni AR runtime subprocess timed out")

    try:
        result = queue.get(timeout=10)
    except Empty:
        pytest.fail(f"Omni AR runtime subprocess exited with code {process.exitcode} without a result")
    assert "error" not in result, result.get("error")
    assert process.exitcode == 0
    return result


def _assert_memory_trace(rank_trace: dict[str, Any]) -> None:
    memory = rank_trace["memory"]
    assert 0 < memory["start_allocated_bytes"] <= memory["peak_allocated_bytes"]
    assert memory["start_allocated_bytes"] <= memory["start_reserved_bytes"]
    assert memory["peak_allocated_bytes"] <= memory["peak_reserved_bytes"]
    assert memory["peak_reserved_bytes"] <= memory["total_bytes"]
    assert 0 < memory["start_free_bytes"] <= memory["total_bytes"]
    assert 0 < memory["end_free_bytes"] <= memory["total_bytes"]
    assert memory["end_allocated_bytes"] <= memory["peak_allocated_bytes"]
    assert memory["end_reserved_bytes"] <= memory["peak_reserved_bytes"]


def test_vibevoice_runtime_scheduler_001(
    omni_ar_processing_result: dict[str, Any],
) -> None:
    result = omni_ar_processing_result
    assert result["scheduler_cls"].endswith(".OmniARAsyncScheduler")
    assert result["worker_type"] == "ar"
    assert result["num_outputs"] >= 1
    assert result["finished"] is True
    assert len(result["token_ids"]) == 1
    assert result["token_ids"][0] in {151652, 151653, 151654, 151643}
    assert result["forced_audio_token_ids"] == [151654, 151654, 151654]
    assert result["forced_finished"] is True


def test_vibevoice_runtime_waveform_002(
    omni_ar_processing_result: dict[str, Any],
) -> None:
    result = omni_ar_processing_result
    # All three sampled audio tokens publish one 3200-sample chunk. The third
    # is emitted by terminal drain because no further AR step is scheduled.
    expected_first_audio_shape = (3_200,) if result["token_ids"][-1] == 151654 else None
    assert result["audio_shape"] == expected_first_audio_shape
    assert result["forced_audio_shape"] == (9_600,)
    assert result["forced_audio_dtype"] == "torch.float32"
    assert result["forced_audio_finite"] is True
    assert result["forced_sample_rate"] == 24_000
    assert [item["rank"] for item in result["forced_runtime_state"]] == [0, 1]
    for rank_state in result["forced_runtime_state"]:
        branch = rank_state["named_branches"]["negative"]
        assert branch["requests"] == {}
        assert branch["num_free_blocks"] == branch["num_blocks"]


def test_vibevoice_runtime_concurrency_003(
    omni_ar_processing_result: dict[str, Any],
) -> None:
    result = omni_ar_processing_result
    for index, request_result in enumerate(result["concurrent_results"]):
        expected_tokens = 10 if index == 0 else 8
        assert request_result["token_ids"] == [151654] * expected_tokens
        assert request_result["finish_reason"] == "length"
        assert request_result["audio_shape"] == (3_200 * expected_tokens,)
        assert request_result["audio_finite"] is True

    assert [item["rank"] for item in result["concurrency_trace"]] == [0, 1]
    for rank_trace in result["concurrency_trace"]:
        _assert_memory_trace(rank_trace)
        assert rank_trace["max_active_requests"] >= 2
        assert any(len(batch) == 2 for batch in rank_trace["negative_batches"]), rank_trace
        assert any(
            item["batch_size"] == 2 and item["noise_shape"] == [4, 64] for item in rank_trace["diffusion_batches"]
        )
        assert any(
            len(batch) == 2
            and any(request_id.startswith("omni-ar-concurrent-a-") for request_id in batch)
            and any(request_id.startswith("omni-ar-concurrent-b-") for request_id in batch)
            for batch in rank_trace["terminal_batches"]
        ), rank_trace
        assert any(
            any(request_id.startswith("omni-ar-concurrent-a-") for request_id in excluded)
            and any(request_id.startswith("omni-ar-concurrent-b-") for request_id in excluded)
            for excluded in rank_trace["cleanup_exclusions"]
        ), rank_trace

    for rank_state in result["concurrent_runtime_state"]:
        branch = rank_state["named_branches"]["negative"]
        assert branch["requests"] == {}
        assert branch["num_free_blocks"] == branch["num_blocks"]


def test_vibevoice_runtime_abort_004(
    omni_ar_processing_result: dict[str, Any],
) -> None:
    result = omni_ar_processing_result
    assert result["abort_cancelled"] is True
    survivor = result["abort_survivor_result"]
    assert survivor["token_ids"] == [151654] * 8
    assert survivor["finish_reason"] == "length"
    assert survivor["audio_shape"] == (25_600,)

    assert [item["rank"] for item in result["abort_trace"]] == [0, 1]
    for rank_trace in result["abort_trace"]:
        _assert_memory_trace(rank_trace)
        assert rank_trace["max_active_requests"] >= 2
        assert any(len(batch) == 2 for batch in rank_trace["negative_batches"])

    for rank_state in result["abort_runtime_state"]:
        branch = rank_state["named_branches"]["negative"]
        assert branch["requests"] == {}
        assert branch["num_free_blocks"] == branch["num_blocks"]


# ======================================================================
# natural-EOS lifecycle (merged from test_vibevoice_natural_eos_gpu.py)
# ======================================================================

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
def test_vibevoice_runtime_natural_eos_005(
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
def test_vibevoice_runtime_transitions_006(
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


def test_vibevoice_runtime_branch_free_007(
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


# ======================================================================
# async-omni cleanup (merged from test_vibevoice_async_omni_cleanup_gpu.py)
# ======================================================================

_WORKER_EXTENSION = "tests.helpers.vibevoice_worker_extension.VibeVoiceWorkerExtensionForTest"


def _all_branch_blocks_released(states: list[dict[str, Any]]) -> bool:
    return bool(states) and all(
        branch["num_free_blocks"] == branch["num_blocks"]
        for state in states
        for branch in state["named_branches"].values()
    )


def _request_ids_matching(
    states: list[dict[str, Any]],
    external_request_id: str,
) -> set[str]:
    return {
        request_id
        for state in states
        for request_id in state["request_states"]
        if request_id.startswith(f"{external_request_id}-")
    }


def _request_has_audio_state(
    states: list[dict[str, Any]],
    external_request_id: str,
) -> bool:
    for state in states:
        for request_id, request_state in state["request_states"].items():
            if not request_id.startswith(f"{external_request_id}-"):
                continue
            branch = state["named_branches"]["negative"]
            negative_state = branch["requests"].get(request_id)
            if (
                request_state["audio_token_count"] >= 1
                and request_state["has_acoustic_cache"]
                and request_state["has_semantic_cache"]
                and negative_state is not None
                and negative_state["num_tokens"] >= 1
            ):
                return True
    return False


def _assert_request_cleaned(
    states: list[dict[str, Any]],
    external_request_id: str,
) -> None:
    assert not _request_ids_matching(states, external_request_id)
    assert all(
        not any(request_id.startswith(f"{external_request_id}-") for request_id in state["deferred_cleanup_ids"])
        for state in states
    )
    assert _all_branch_blocks_released(states)


def _async_omni_cleanup_worker(
    model_path: str,
    tokenizer_path: str,
    deploy_path: str,
    report_dir: str,
    queue: Any,
) -> None:
    try:
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        import numpy as np
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        from vllm_omni.entrypoints.async_omni import AsyncOmni

        async def run() -> dict[str, Any]:
            engine = AsyncOmni(
                model=model_path,
                tokenizer=tokenizer_path,
                deploy_config=deploy_path,
                stage_overrides={
                    "0": {
                        "worker_extension_cls": _WORKER_EXTENSION,
                    }
                },
                stage_init_timeout=300,
                init_timeout=600,
                log_stats=False,
            )
            shutdown_complete = False
            try:
                tokenizer = engine.engine.input_processor.renderer.get_tokenizer()
                prompt_text = "<|vision_start|><|vision_pad|><|vision_end|> Speech output:\n<|vision_start|>"

                def make_prompt(request_id: str) -> dict[str, Any]:
                    return {
                        "prompt": prompt_text,
                        "prompt_token_ids": tokenizer.encode(
                            prompt_text,
                            add_special_tokens=False,
                        ),
                        "multi_modal_data": {"audio": [(np.zeros(3_200, dtype=np.float32), 24_000)]},
                        "multi_modal_uuids": {"audio": [f"{request_id}:audio:0"]},
                    }

                def sampling(
                    max_tokens: int,
                    *,
                    allowed_token_ids: list[int] | None = None,
                ) -> SamplingParams:
                    return SamplingParams(
                        max_tokens=max_tokens,
                        temperature=0.0,
                        allowed_token_ids=allowed_token_ids or [151654],
                        stop_token_ids=[151643],
                        detokenize=False,
                        output_kind=RequestOutputKind.DELTA,
                        extra_args={
                            "guidance_scale": 1.3,
                            "num_diffusion_steps": 1,
                        },
                    )

                async def runtime_state() -> list[dict[str, Any]]:
                    result = await engine.collective_rpc(
                        method="vibevoice_test_runtime_state",
                        timeout=60,
                        stage_ids=[0],
                    )
                    assert len(result) == 1
                    states = result[0]
                    assert len(states) == 2
                    return sorted(states, key=lambda item: item["rank"])

                async def run_to_completion(
                    request_id: str,
                    *,
                    params: SamplingParams | None = None,
                ) -> list[Any]:
                    outputs = []
                    async for output in engine.generate(
                        prompt=make_prompt(request_id),
                        request_id=request_id,
                        sampling_params_list=[params or sampling(3)],
                        output_modalities=["audio"],
                    ):
                        outputs.append(output)
                    return outputs

                async def wait_for_audio_state(request_id: str) -> list[dict[str, Any]]:
                    deadline = time.monotonic() + 120
                    while time.monotonic() < deadline:
                        states = await runtime_state()
                        if _request_has_audio_state(states, request_id):
                            return states
                        await asyncio.sleep(0.05)
                    raise AssertionError(f"VibeVoice request {request_id!r} never reached audio state")

                initial = await runtime_state()
                assert _all_branch_blocks_released(initial)
                assert all(not branch["requests"] for state in initial for branch in state["named_branches"].values())

                finish_outputs = await run_to_completion("finish")
                assert finish_outputs
                finish_token_ids = [token_id for output in finish_outputs for token_id in output.outputs[0].token_ids]
                finish_audio_chunks = [
                    audio
                    for output in finish_outputs
                    if isinstance(
                        audio := output.multimodal_output.get("audio"),
                        torch.Tensor,
                    )
                ]
                assert finish_token_ids == [151654, 151654, 151654]
                assert sum(int(audio.numel()) for audio in finish_audio_chunks) == 6_400

                finish_before_safe_point = await runtime_state()
                assert _request_ids_matching(
                    finish_before_safe_point,
                    "finish",
                )

                finish_safe_outputs = await run_to_completion(
                    "safe-after-finish",
                    params=sampling(1, allowed_token_ids=[151643]),
                )
                assert finish_safe_outputs
                finish_after_safe_point = await runtime_state()
                _assert_request_cleaned(finish_after_safe_point, "finish")

                abort_outputs: list[Any] = []

                async def run_abort_request() -> None:
                    async for output in engine.generate(
                        prompt=make_prompt("abort"),
                        request_id="abort",
                        sampling_params_list=[sampling(100)],
                        output_modalities=["audio"],
                    ):
                        abort_outputs.append(output)

                abort_task = asyncio.create_task(run_abort_request())
                abort_active = await wait_for_audio_state("abort")
                abort_internal_ids = _request_ids_matching(
                    abort_active,
                    "abort",
                )
                assert len(abort_internal_ids) == 1
                await engine.abort("abort")
                abort_task.cancel()
                try:
                    await abort_task
                except asyncio.CancelledError:
                    pass

                abort_before_safe_point = await runtime_state()
                assert _request_ids_matching(abort_before_safe_point, "abort")
                abort_output_count = len(abort_outputs)
                await asyncio.sleep(0.1)
                assert len(abort_outputs) == abort_output_count

                abort_safe_outputs = await run_to_completion(
                    "safe-after-abort",
                    params=sampling(1, allowed_token_ids=[151643]),
                )
                assert abort_safe_outputs
                abort_after_safe_point = await runtime_state()
                _assert_request_cleaned(abort_after_safe_point, "abort")

                armed = await engine.collective_rpc(
                    method="vibevoice_test_arm_negative_fault",
                    timeout=60,
                    args=(2, report_dir),
                    stage_ids=[0],
                )
                assert len(armed) == 1
                assert [item["rank"] for item in armed[0]] == [0, 1]
                assert all(item["armed"] for item in armed[0])

                exception_text = None
                try:
                    async for _ in engine.generate(
                        prompt=make_prompt("exception"),
                        request_id="exception",
                        sampling_params_list=[sampling(4)],
                        output_modalities=["audio"],
                    ):
                        pass
                except Exception as exc:
                    exception_text = str(exc)
                assert exception_text is not None

                engine.shutdown()
                shutdown_complete = True

                report_paths = [Path(report_dir) / f"rank-{rank}.json" for rank in range(2)]
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline and not all(path.is_file() for path in report_paths):
                    await asyncio.sleep(0.05)
                assert all(path.is_file() for path in report_paths)
                shutdown_reports = [json.loads(path.read_text()) for path in report_paths]
                for report in shutdown_reports:
                    pre_shutdown = report["pre_shutdown"]
                    exception_ids = [
                        request_id
                        for request_id in pre_shutdown["request_states"]
                        if request_id.startswith("exception-")
                    ]
                    assert len(exception_ids) == 1
                    request_id = exception_ids[0]
                    negative = pre_shutdown["named_branches"]["negative"]
                    assert request_id not in negative["requests"]
                    assert negative["num_free_blocks"] == negative["num_blocks"]

                    post_shutdown = report["post_shutdown"]
                    assert post_shutdown == {
                        "model_collected": True,
                        "runner_named_branches_empty": True,
                        "branch_closed": True,
                        "branch_entered": False,
                        "branch_request_ids": [],
                        "num_blocks": post_shutdown["num_blocks"],
                        "num_free_blocks": post_shutdown["num_blocks"],
                        "kv_caches_empty": True,
                        "raw_caches_empty": True,
                    }

                return {
                    "initial_released": _all_branch_blocks_released(initial),
                    "finish_before_safe_point_ids": sorted(_request_ids_matching(finish_before_safe_point, "finish")),
                    "finish_cleaned": not _request_ids_matching(
                        finish_after_safe_point,
                        "finish",
                    ),
                    "abort_internal_ids": sorted(abort_internal_ids),
                    "abort_before_safe_point_ids": sorted(_request_ids_matching(abort_before_safe_point, "abort")),
                    "abort_cleaned": not _request_ids_matching(
                        abort_after_safe_point,
                        "abort",
                    ),
                    "abort_output_count": abort_output_count,
                    "exception_text": exception_text,
                    "shutdown_report_count": len(shutdown_reports),
                }
            finally:
                if not shutdown_complete:
                    engine.shutdown()

        queue.put(asyncio.run(run()))
    except Exception:
        queue.put({"error": traceback.format_exc()})


def test_vibevoice_runtime_cleanup_008() -> None:
    model_root = os.getenv("VIBEVOICE_TEST_MODEL_ROOT")
    if not model_root:
        pytest.skip("Set VIBEVOICE_TEST_MODEL_ROOT to run AsyncOmni cleanup coverage")

    model_path = Path(model_root) / "VibeVoice"
    tokenizer_path = Path(model_root) / "VibeVoice-1.5B-hf"
    deploy_path = Path(__file__).parents[4] / "vllm_omni/deploy/vibevoice.yaml"
    if not (model_path / "model.safetensors.index.json").is_file():
        pytest.fail(f"Official VibeVoice checkpoint not found at {model_path}")
    if not (tokenizer_path / "tokenizer.json").is_file():
        pytest.skip(f"VibeVoice tokenizer fixture not found at {tokenizer_path}")

    context = mp.get_context("spawn")
    queue = context.Queue()
    with tempfile.TemporaryDirectory(prefix="vibevoice-shutdown-") as report_dir:
        process = context.Process(
            target=_async_omni_cleanup_worker,
            args=(
                str(model_path),
                str(tokenizer_path),
                str(deploy_path),
                report_dir,
                queue,
            ),
        )
        process.start()
        process.join(timeout=600)
        if process.is_alive():
            process.kill()
            process.join()
            pytest.fail("VibeVoice AsyncOmni cleanup subprocess timed out")

        try:
            result = queue.get(timeout=10)
        except Empty:
            pytest.fail(f"AsyncOmni cleanup subprocess exited without a result: exitcode={process.exitcode}")

    assert "error" not in result, result.get("error")
    assert process.exitcode == 0
    assert result["initial_released"] is True
    assert len(result["finish_before_safe_point_ids"]) == 1
    assert result["finish_cleaned"] is True
    assert len(result["abort_internal_ids"]) == 1
    assert result["abort_before_safe_point_ids"] == result["abort_internal_ids"]
    assert result["abort_cleaned"] is True
    assert result["abort_output_count"] >= 0
    assert result["exception_text"]
    assert result["shutdown_report_count"] == 2


# ======================================================================
# engine-core lifecycle (merged from test_vibevoice_engine_core_gpu.py)
# ======================================================================


def _engine_core_worker(model_path_str: str, tokenizer_path_str: str, queue: Any) -> None:
    try:
        # Keep EngineCore and the GPU runner in this subprocess so the test can
        # inspect the real scheduler/runner cache transition directly.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        # The test environment does not require FlashInfer's JIT sampler and may
        # not provide the external ninja binary.
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

        import numpy as np
        from vllm import LLM, SamplingParams
        from vllm.v1.worker.gpu_input_batch import InputBatch
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        from vllm_omni.engine.arg_utils import register_omni_models_to_vllm
        from vllm_omni.model_executor.models.vibevoice.pipeline import (
            VIBEVOICE_VALID_TOKEN_IDS,
        )
        from vllm_omni.model_executor.models.vibevoice.vibevoice import (
            VibeVoiceForConditionalGeneration,
        )

        register_omni_models_to_vllm()

        freed_notifications: list[str] = []
        original_update_states = GPUModelRunner._update_states

        def record_freed_hashes(self, scheduler_output):
            freed_notifications.extend(scheduler_output.free_encoder_mm_hashes)
            return original_update_states(self, scheduler_output)

        GPUModelRunner._update_states = record_freed_hashes

        encoder_calls = 0
        original_audio_embeddings = VibeVoiceForConditionalGeneration._get_audio_embeddings

        def count_audio_embeddings(self, *args, **kwargs):
            nonlocal encoder_calls
            encoder_calls += 1
            return original_audio_embeddings(self, *args, **kwargs)

        VibeVoiceForConditionalGeneration._get_audio_embeddings = count_audio_embeddings

        observed_allowed_sets: list[list[int]] = []
        original_make_sampling_metadata = InputBatch._make_sampling_metadata

        def record_sampling_metadata(self):
            metadata = original_make_sampling_metadata(self)
            mask = metadata.allowed_token_ids_mask
            if mask is not None and mask.shape[0] > 0:
                allowed = (~mask[0]).nonzero(as_tuple=False).flatten().cpu().tolist()
                observed_allowed_sets.append(allowed)
            return metadata

        InputBatch._make_sampling_metadata = record_sampling_metadata

        llm = LLM(
            model=model_path_str,
            tokenizer=tokenizer_path_str,
            trust_remote_code=False,
            dtype="bfloat16",
            max_model_len=1024,
            max_num_batched_tokens=450,
            max_num_seqs=1,
            kv_cache_memory_bytes=1024**3,
            gpu_memory_utilization=0.2,
            enforce_eager=True,
            enable_prefix_caching=False,
            limit_mm_per_prompt={"audio": 1},
            # Must be non-zero or pinned vLLM intentionally replaces caller
            # UUIDs with one-shot renderer IDs and disables cross-request reuse.
            mm_processor_cache_gb=0.1,
            # Pinned v1's generic text-only MM profiler conflicts with the
            # model's deliberate symmetric placeholder validation.
            skip_mm_profiling=True,
        )
        try:
            waveform = np.zeros(1_440_000, dtype=np.float32)  # 60 s -> 450 tokens
            prompt_text = "<|vision_start|><|vision_pad|><|vision_end|> hello"
            prompt_token_ids = llm.get_tokenizer().encode(
                prompt_text,
                add_special_tokens=False,
            )

            def prompt(mm_uuid: str):
                return {
                    "prompt": prompt_text,
                    # The token path is intentional: pinned vLLM drops MM UUIDs
                    # from its text-prompt path.
                    "prompt_token_ids": prompt_token_ids,
                    "multi_modal_data": {"audio": [(waveform, 24_000)]},
                    "multi_modal_uuids": {"audio": [mm_uuid]},
                }

            params = SamplingParams(
                max_tokens=1,
                temperature=0.0,
                allowed_token_ids=list(VIBEVOICE_VALID_TOKEN_IDS),
                stop_token_ids=[151643],
                detokenize=False,
            )
            assert params.all_stop_token_ids == {151643}

            core = llm.llm_engine.engine_core.engine_core
            manager = core.scheduler.encoder_cache_manager
            runner = core.model_executor.driver_worker.worker.model_runner

            llm.generate(prompt("same-uuid"), params, use_tqdm=False)
            calls_after_first = encoder_calls
            finish_freeable = (
                manager.cached.get("same-uuid") == set()
                and "same-uuid" in manager.freeable
                and "same-uuid" in runner.encoder_cache
            )

            llm.generate(prompt("same-uuid"), params, use_tqdm=False)
            calls_after_same = encoder_calls

            llm.generate(prompt("different-uuid"), params, use_tqdm=False)
            calls_after_different = encoder_calls
            finish_eviction = (
                "same-uuid" in freed_notifications
                and "same-uuid" not in manager.cached
                and "same-uuid" not in runner.encoder_cache
                and "different-uuid" in runner.encoder_cache
            )

            # Abort while the 450-token placeholder is still at the chunked
            # prefill frontier, so abort itself must release the active cache
            # reference and make the entry freeable.
            abort_params = SamplingParams(
                max_tokens=100,
                temperature=0.0,
                allowed_token_ids=[151654],
                stop_token_ids=[151643],
                detokenize=False,
            )
            abort_calls_before = encoder_calls
            llm.llm_engine.add_request(
                "abort-request",
                prompt("abort-uuid"),
                abort_params,
            )
            for _ in range(3):
                llm.llm_engine.step()
                if encoder_calls > abort_calls_before:
                    break
            abort_was_encoded = encoder_calls == abort_calls_before + 1
            llm.llm_engine.abort_request(["abort-request"])
            abort_freeable = (
                manager.cached.get("abort-uuid") == set()
                and "abort-uuid" in manager.freeable
                and "abort-uuid" in runner.encoder_cache
            )

            llm.generate(prompt("post-abort-pressure"), params, use_tqdm=False)
            abort_eviction = (
                "abort-uuid" in freed_notifications
                and "abort-uuid" not in manager.cached
                and "abort-uuid" not in runner.encoder_cache
            )

            queue.put(
                {
                    "calls_after_first": calls_after_first,
                    "calls_after_same": calls_after_same,
                    "calls_after_different": calls_after_different,
                    "finish_freeable": finish_freeable,
                    "finish_eviction": finish_eviction,
                    "abort_was_encoded": abort_was_encoded,
                    "abort_freeable": abort_freeable,
                    "abort_eviction": abort_eviction,
                    "saw_four_token_metadata": sorted(VIBEVOICE_VALID_TOKEN_IDS)
                    in [sorted(ids) for ids in observed_allowed_sets],
                }
            )
        finally:
            llm.llm_engine.engine_core.shutdown()
    except Exception:
        queue.put({"error": traceback.format_exc()})


def test_vibevoice_runtime_engine_core_009() -> None:
    model_root = os.getenv("VIBEVOICE_TEST_MODEL_ROOT")
    if not model_root:
        pytest.skip("Set VIBEVOICE_TEST_MODEL_ROOT to run real EngineCore coverage")

    model_path = Path(model_root) / "VibeVoice"
    tokenizer_path = Path(model_root) / "VibeVoice-1.5B-hf"
    if not (model_path / "model.safetensors.index.json").is_file():
        pytest.fail(f"Official VibeVoice checkpoint not found at {model_path}")
    if not (tokenizer_path / "tokenizer.json").is_file():
        pytest.skip(f"VibeVoice tokenizer fixture not found at {tokenizer_path}")

    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_engine_core_worker,
        args=(str(model_path), str(tokenizer_path), queue),
    )
    process.start()
    process.join(timeout=360)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("VibeVoice EngineCore subprocess timed out")

    try:
        result = queue.get(timeout=5)
    except Empty:
        pytest.fail(f"EngineCore subprocess exited with code {process.exitcode} without a result")
    assert "error" not in result, result.get("error")
    assert process.exitcode == 0
    assert result == {
        "calls_after_first": 1,
        "calls_after_same": 1,
        "calls_after_different": 2,
        "finish_freeable": True,
        "finish_eviction": True,
        "abort_was_encoded": True,
        "abort_freeable": True,
        "abort_eviction": True,
        "saw_four_token_metadata": True,
    }
