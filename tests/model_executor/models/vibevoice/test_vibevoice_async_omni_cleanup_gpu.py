# SPDX-License-Identifier: Apache-2.0
"""Real AsyncOmni cleanup coverage for VibeVoice side state."""

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

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

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


def test_real_async_omni_reclaims_state_after_finish_abort_and_exception() -> None:
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
