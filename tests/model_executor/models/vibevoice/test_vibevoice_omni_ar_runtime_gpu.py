# SPDX-License-Identifier: Apache-2.0
"""Opt-in Processing smoke through the real Omni AR runtime.

This intentionally generates one control token only. Stateful waveform
inference is outside this test's scope.
"""

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
                stage_init_timeout=300,
                init_timeout=600,
                log_stats=False,
            )
            try:
                stage = engine.stage_configs[0]
                scheduler_cls = str(stage.engine_args.scheduler_cls)
                worker_type = str(stage.engine_args.worker_type)

                tokenizer = engine.engine.input_processor.renderer.get_tokenizer()
                prompt_text = (
                    "<|vision_start|><|vision_pad|><|vision_end|> "
                    "Speech output:\n<|vision_start|>"
                )
                prompt = {
                    "prompt": prompt_text,
                    "prompt_token_ids": tokenizer.encode(
                        prompt_text,
                        add_special_tokens=False,
                    ),
                    "multi_modal_data": {
                        "audio": [(np.zeros(3_200, dtype=np.float32), 24_000)]
                    },
                    "multi_modal_uuids": {
                        "audio": ["omni-ar-processing-smoke:audio:0"]
                    },
                }
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
                return {
                    "scheduler_cls": scheduler_cls,
                    "worker_type": worker_type,
                    "num_outputs": len(outputs),
                    "finished": bool(final.finished),
                    "token_ids": token_ids,
                    "audio_shape": tuple(audio.shape) if isinstance(audio, torch.Tensor) else None,
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
        pytest.fail(
            f"Omni AR runtime subprocess exited with code {process.exitcode} without a result"
        )
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known pre-inference boundary: final_output_type=audio currently exposes "
        "reference encoder embeddings rather than a decoded 24 kHz waveform"
    ),
)
def test_processing_runtime_does_not_mislabel_reference_embeddings_as_waveform(
    omni_ar_processing_result: dict[str, Any],
) -> None:
    audio_shape = omni_ar_processing_result["audio_shape"]
    assert audio_shape is None or len(audio_shape) == 1
