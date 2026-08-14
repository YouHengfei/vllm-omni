# SPDX-License-Identifier: Apache-2.0
"""Opt-in waveform smoke through the real Omni AR runtime."""

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


def test_processing_runs_on_official_omni_ar_async_scheduler(
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


def test_runtime_publishes_decoded_mono_24khz_waveform(
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


def test_runtime_executes_two_resident_requests_with_batched_terminal_rng(
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


def test_aborting_one_resident_request_does_not_interrupt_the_other(
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
