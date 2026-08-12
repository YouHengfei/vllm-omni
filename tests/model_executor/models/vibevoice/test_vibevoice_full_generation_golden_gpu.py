# SPDX-License-Identifier: Apache-2.0
"""Cached full-generation parity with Transformers PR #40546."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
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
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Two CUDA devices are required",
    ),
]

_WORKER_EXTENSION = "tests.helpers.vibevoice_worker_extension.VibeVoiceWorkerExtensionForTest"
_GOLDEN_SEED = 12_345
_TRACE_KEYS = (
    "negative_input_embedding",
    "positive_condition",
    "negative_condition",
    "noise",
    "audio_latent",
    "audio",
    "semantic_latent",
    "next_embedding",
)


def _run_omni_generation(
    model_path: str,
    tokenizer_path: str,
    deploy_path: str,
    output_dir: str,
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
                stage_overrides={"0": {"worker_extension_cls": _WORKER_EXTENSION}},
                stage_init_timeout=300,
                init_timeout=600,
                log_stats=False,
            )
            try:
                armed = await engine.collective_rpc(
                    method="vibevoice_test_arm_generation_trace",
                    args=(_GOLDEN_SEED,),
                    timeout=60,
                    stage_ids=[0],
                )
                assert len(armed) == 1
                assert [item["rank"] for item in armed[0]] == [0, 1]
                assert all(item["armed"] for item in armed[0])

                tokenizer = engine.engine.input_processor.renderer.get_tokenizer()
                prompt_text = "<|vision_start|><|vision_pad|><|vision_end|> Speech output:\n<|vision_start|>"
                prompt = {
                    "prompt": prompt_text,
                    "prompt_token_ids": tokenizer.encode(
                        prompt_text,
                        add_special_tokens=False,
                    ),
                    "multi_modal_data": {
                        "audio": [
                            (
                                np.linspace(
                                    -0.2,
                                    0.2,
                                    3_200,
                                    dtype=np.float32,
                                ),
                                24_000,
                            )
                        ]
                    },
                    "multi_modal_uuids": {"audio": ["full-generation-golden:audio:0"]},
                }
                params = SamplingParams(
                    max_tokens=3,
                    temperature=0.0,
                    allowed_token_ids=[151654],
                    stop_token_ids=[151643],
                    detokenize=False,
                    output_kind=RequestOutputKind.DELTA,
                    extra_args={
                        "guidance_scale": 1.3,
                        "num_diffusion_steps": 1,
                    },
                )
                outputs = []
                async for output in engine.generate(
                    prompt=prompt,
                    request_id="full-generation-golden",
                    sampling_params_list=[params],
                    output_modalities=["audio"],
                ):
                    outputs.append(output)
                token_ids = [token_id for output in outputs for token_id in output.outputs[0].token_ids]
                audio_chunks = [
                    audio
                    for output in outputs
                    if isinstance(
                        audio := output.multimodal_output.get("audio"),
                        torch.Tensor,
                    )
                ]
                sample_rates = []
                for output in outputs:
                    sample_rate = output.multimodal_output.get("sr")
                    if sample_rate is None:
                        continue
                    values = sample_rate if isinstance(sample_rate, list) else [sample_rate]
                    sample_rates.extend(
                        int(value.item()) if isinstance(value, torch.Tensor) else int(value) for value in values
                    )
                written = await engine.collective_rpc(
                    method="vibevoice_test_write_generation_trace",
                    args=(output_dir,),
                    timeout=60,
                    stage_ids=[0],
                )
                assert len(written) == 1
                assert [item["rank"] for item in written[0]] == [0, 1]
                assert all(item["num_steps"] == 2 for item in written[0])
                audio_path = Path(output_dir) / "omni-audio.pt"
                torch.save(torch.cat(audio_chunks).contiguous(), audio_path)
                return {
                    "token_ids": token_ids,
                    "audio_path": str(audio_path),
                    "sample_rates": sample_rates,
                }
            finally:
                engine.shutdown()

        queue.put(asyncio.run(run()))
    except Exception:
        queue.put({"error": traceback.format_exc()})


def _compare_trace_tensor(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
) -> None:
    assert tuple(actual.shape) == tuple(expected.shape), name
    assert torch.isfinite(actual).all(), name
    assert torch.isfinite(expected).all(), name
    tolerances = {
        "audio_latent": (0.05, 0.05),
        "audio": (2e-5, 0.05),
        # Cached BF16 convolutions are backend/shape dependent; the existing
        # M4b contract uses the same 0.25 bounded parity guard.
        "semantic_latent": (0.25, 0.05),
        "next_embedding": (0.25, 0.05),
    }
    atol, rtol = tolerances[name]
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=atol,
        rtol=rtol,
        msg=name,
    )


def test_full_cached_generation_matches_transformers_pr_reference() -> None:
    model_root = os.getenv("VIBEVOICE_TEST_MODEL_ROOT")
    transformers_root = os.getenv("VIBEVOICE_TRANSFORMERS_PR_ROOT")
    if not model_root:
        pytest.skip("Set VIBEVOICE_TEST_MODEL_ROOT to run full generation golden")
    if not transformers_root:
        pytest.skip("Set VIBEVOICE_TRANSFORMERS_PR_ROOT to a Transformers PR #40546 checkout")

    model_path = Path(model_root) / "VibeVoice"
    tokenizer_path = Path(model_root) / "VibeVoice-1.5B-hf"
    transformers_path = Path(transformers_root)
    deploy_path = Path(__file__).parents[4] / "vllm_omni/deploy/vibevoice.yaml"
    hf_helper = Path(__file__).parents[3] / "helpers/vibevoice_hf_golden.py"
    if not (model_path / "model.safetensors.index.json").is_file():
        pytest.fail(f"Official VibeVoice checkpoint not found at {model_path}")
    if not (tokenizer_path / "tokenizer.json").is_file():
        pytest.skip(f"VibeVoice tokenizer fixture not found at {tokenizer_path}")
    if not (transformers_path / "src/transformers/models/vibevoice/generation_vibevoice.py").is_file():
        pytest.skip(f"Transformers VibeVoice PR checkout not found at {transformers_path}")

    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="vibevoice-full-golden-") as output_dir:
        output_path = Path(output_dir)
        hf_output_path = output_path / "transformers.pt"
        queue = context.Queue()
        process = context.Process(
            target=_run_omni_generation,
            args=(
                str(model_path),
                str(tokenizer_path),
                str(deploy_path),
                output_dir,
                queue,
            ),
        )
        process.start()
        process.join(timeout=600)
        if process.is_alive():
            process.kill()
            process.join()
            pytest.fail("VibeVoice Omni full-generation golden timed out")
        try:
            omni_result = queue.get(timeout=10)
        except Empty:
            pytest.fail(
                f"VibeVoice Omni full-generation subprocess exited without a result: exitcode={process.exitcode}"
            )
        assert "error" not in omni_result, omni_result.get("error")
        assert process.exitcode == 0
        omni_audio = torch.load(
            omni_result["audio_path"],
            map_location="cpu",
            weights_only=True,
        )
        omni_trace_payloads = [
            torch.load(
                output_path / f"omni-rank-{rank}.pt",
                map_location="cpu",
                weights_only=True,
            )
            for rank in range(2)
        ]
        omni_traces = [payload["steps"] for payload in omni_trace_payloads]

        hf_environment = os.environ.copy()
        hf_environment["PYTHONPATH"] = str(transformers_path / "src")
        subprocess.run(
            [
                sys.executable,
                str(hf_helper),
                "--checkpoint",
                str(model_path),
                "--runtime-schema",
                str(tokenizer_path),
                "--output",
                str(hf_output_path),
                "--omni-trace",
                str(output_path / "omni-rank-0.pt"),
                "--seed",
                str(_GOLDEN_SEED),
            ],
            env=hf_environment,
            check=True,
            timeout=600,
        )
        hf_result = torch.load(
            hf_output_path,
            map_location="cpu",
            weights_only=True,
        )

    assert hf_result["token_ids"] == [151654, 151654]
    assert omni_result["token_ids"] == [151654, 151654, 151654]
    assert hf_result["audio"].shape == (1, 6_400)
    assert omni_audio.shape == (6_400,)
    assert omni_audio.dtype == torch.float32
    assert omni_result["sample_rates"] and set(omni_result["sample_rates"]) == {24_000}
    assert torch.isfinite(omni_audio).all()
    assert len(hf_result["trace"]) == 2
    assert [len(trace) for trace in omni_traces] == [2, 2]
    assert len(hf_result["encoded_reference_latents"]) == 1
    assert [len(payload["encoded_reference_latents"]) for payload in omni_trace_payloads] == [1, 1]
    for payload in omni_trace_payloads:
        torch.testing.assert_close(
            payload["encoded_reference_latents"][0].float(),
            hf_result["encoded_reference_latents"][0].float(),
            atol=0.03,
            rtol=0.03,
        )

    for rank_trace in omni_traces[1:]:
        for step_index, (actual_step, rank_zero_step) in enumerate(zip(rank_trace, omni_traces[0], strict=True)):
            for key in _TRACE_KEYS:
                torch.testing.assert_close(
                    actual_step[key],
                    rank_zero_step[key],
                    msg=f"TP rank parity: step={step_index}, tensor={key}",
                )

    assert len(hf_result["replay_trace"]) == 2
    replay_keys = (
        "audio_latent",
        "audio",
        "semantic_latent",
        "next_embedding",
    )
    for step_index, (omni_step, hf_step) in enumerate(
        zip(
            omni_traces[0],
            hf_result["replay_trace"],
            strict=True,
        )
    ):
        for key in replay_keys:
            _compare_trace_tensor(
                omni_step[key],
                hf_step[key],
                name=key,
            )
