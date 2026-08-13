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
