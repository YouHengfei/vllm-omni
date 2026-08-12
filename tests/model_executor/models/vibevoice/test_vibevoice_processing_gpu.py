# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in GPU test for VibeVoice Processor-to-prefill integration."""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
import traceback
from contextlib import contextmanager
from pathlib import Path
from queue import Empty
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _gpu_processing_worker(
    model_path_str: str,
    tokenizer_path_str: str,
    port: int,
    queue: Any,
) -> None:
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        )

        import gc

        from transformers import AutoTokenizer
        from vllm.config import set_current_vllm_config
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.model_executor.model_loader import get_model_loader
        from vllm.multimodal.processing import InputProcessingContext

        from vllm_omni.engine.arg_utils import OmniEngineArgs
        from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
        from vllm_omni.model_executor.models.vibevoice.processing_vibevoice import (
            AUDIO_BOS_TOKEN,
            AUDIO_EOS_TOKEN,
            AUDIO_TOKEN,
            SAMPLE_RATE,
        )

        model_path = Path(model_path_str)
        torch.cuda.set_device(0)
        args = OmniEngineArgs(
            model=str(model_path),
            tokenizer=tokenizer_path_str,
            model_arch="VibeVoiceForConditionalGeneration",
            model_stage="latent_generator",
            worker_type="ar",
            dtype="bfloat16",
            load_format="safetensors",
            trust_remote_code=False,
            max_model_len=4096,
            limit_mm_per_prompt={"audio": 8},
            enforce_eager=True,
        )
        config = args.create_engine_config()
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="nccl",
        )
        model = None
        try:
            with set_current_vllm_config(config):
                initialize_model_parallel(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                )
                model = get_model_loader(config.load_config).load_model(
                    vllm_config=config,
                    model_config=config.model_config,
                )

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path_str,
                trust_remote_code=False,
            )
            factories = type(model)._processor_factory
            ctx = InputProcessingContext(config.model_config, tokenizer=tokenizer)
            info = factories.info(ctx)
            processor = factories.build_processor(ctx, cache=None)

            prompt = (
                f"Speaker 0: {AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN} "
                f"Speaker 1: {AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN}"
            )
            first_audio = np.linspace(-0.25, 0.25, 3_201, dtype=np.float32)
            second_audio = np.sin(
                np.linspace(0, 20 * np.pi, 8_001, dtype=np.float32)
            ).astype(np.float32)
            mm_items = info.parse_mm_data(
                {
                    "audio": [
                        (first_audio, SAMPLE_RATE),
                        (second_audio, SAMPLE_RATE),
                    ]
                }
            )
            processed = processor(prompt, mm_items=mm_items)
            mm_data = processed["mm_kwargs"].get_data()
            counts = mm_data["audio_num_tokens"].reshape(-1)
            assert counts.tolist() == [2, 3]
            assert [
                placeholder.length
                for placeholder in processed["mm_placeholders"]["audio"]
            ] == [2, 3]

            input_values = mm_data["input_values"].to("cuda")
            padding_mask = mm_data["padding_mask"].to("cuda")
            counts_cuda = counts.to("cuda")

            # Deterministic parity against the PR #40546 formula.
            deterministic = model._get_audio_embeddings(
                input_values,
                padding_mask,
                counts_cuda,
                sample=False,
            )
            with torch.inference_mode():
                latents = model.model.audio_tower.encode(
                    input_values.to(dtype=torch.bfloat16),
                    sample=False,
                ).latents
                features = (
                    latents + model.model.latent_bias_factor.to(latents.device)
                ) * model.model.latent_scaling_factor.to(latents.device)
                projected = model.model.multi_modal_projector(features)
            placeholders = processed["mm_placeholders"]["audio"]
            for item_idx, num_tokens in enumerate((2, 3)):
                assert deterministic[item_idx].shape == (num_tokens, 1536)
                assert deterministic[item_idx].shape[0] == placeholders[item_idx].length
                torch.testing.assert_close(
                    deterministic[item_idx],
                    projected[item_idx, :num_tokens],
                    rtol=0,
                    atol=0,
                )

            # sample=True parity requires resetting the RNG before each path.
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            sampled = model._get_audio_embeddings(
                input_values,
                padding_mask,
                counts_cuda,
                sample=True,
            )
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            with torch.inference_mode():
                sampled_latents = model.model.audio_tower.encode(
                    input_values.to(dtype=torch.bfloat16),
                    sample=True,
                ).latents
                sampled_features = (
                    sampled_latents
                    + model.model.latent_bias_factor.to(sampled_latents.device)
                ) * model.model.latent_scaling_factor.to(sampled_latents.device)
                sampled_projected = model.model.multi_modal_projector(
                    sampled_features
                )
            for item_idx, num_tokens in enumerate((2, 3)):
                torch.testing.assert_close(
                    sampled[item_idx],
                    sampled_projected[item_idx, :num_tokens],
                    rtol=0,
                    atol=0,
                )

            torch.manual_seed(5678)
            torch.cuda.manual_seed_all(5678)
            resampled = model._get_audio_embeddings(
                input_values,
                padding_mask,
                counts_cuda,
                sample=True,
            )
            assert not torch.equal(sampled[0], resampled[0])

            prompt_ids = torch.tensor(
                processed["prompt_token_ids"],
                device="cuda",
                dtype=torch.long,
            )
            audio_token_id = int(config.model_config.hf_config.audio_token_id)
            is_multimodal = prompt_ids == audio_token_id
            merged = model.embed_input_ids(
                prompt_ids,
                deterministic,
                is_multimodal,
            )
            text_only = model.embed_input_ids(prompt_ids)
            expected_mm = torch.cat(deterministic, dim=0)
            assert int(is_multimodal.sum().item()) == 5
            torch.testing.assert_close(
                merged[is_multimodal],
                expected_mm,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                merged[~is_multimodal],
                text_only[~is_multimodal],
                rtol=0,
                atol=0,
            )
            assert torch.isfinite(merged).all()

            # Drive the real Omni runner preprocessing method with Processor
            # output and the loaded model. Scheduler/cache plumbing is reduced
            # to deterministic stubs, while encoder, scale/bias, projector and
            # merge execute through their production methods.
            runner = object.__new__(OmniGPUModelRunner)
            runner.model = model
            runner.model_config = config.model_config
            runner.vllm_config = config
            runner.supports_mm_inputs = True
            runner.enable_prompt_embeds = False
            runner.uses_mrope = False
            runner.uses_xdrope_dim = 0
            runner.has_talker_mtp = False
            runner.encoder_cache = {}
            runner.input_ids = SimpleNamespace(gpu=prompt_ids.clone())
            runner.inputs_embeds = SimpleNamespace(
                gpu=torch.zeros_like(text_only)
            )
            runner.positions = torch.arange(
                prompt_ids.numel(),
                device="cuda",
                dtype=torch.long,
            )
            runner.input_batch = SimpleNamespace(
                req_ids=["runner-request"],
                num_computed_tokens_cpu=np.array([0], dtype=np.int32),
            )
            runner.query_start_loc = SimpleNamespace(
                cpu=np.array([0, prompt_ids.numel()], dtype=np.int32)
            )
            runner.requests = {}
            runner.model_intermediate_buffer = {}

            @contextmanager
            def encoder_connector_context(*args, **kwargs):
                yield None

            runner.maybe_get_ec_connector_output = encoder_connector_context
            runner_embeddings: list[torch.Tensor] = []

            def execute_encoder(scheduler_output):
                torch.manual_seed(2468)
                torch.cuda.manual_seed_all(2468)
                runner_embeddings.extend(
                    model.embed_multimodal(
                        input_values=input_values,
                        padding_mask=padding_mask,
                        audio_num_tokens=counts_cuda,
                    )
                )
                return runner_embeddings

            runner._execute_mm_encoder = execute_encoder
            runner._gather_mm_embeddings = lambda scheduler_output: (
                runner_embeddings,
                is_multimodal,
            )
            runner._init_model_kwargs = lambda: {}
            runner._extract_mm_kwargs = lambda scheduler_output: {}
            runner._collect_additional_information_for_prefill = lambda counts: None
            runner._update_additional_information = lambda scheduler_output: None
            runner._sync_local_stage_payloads = lambda: None

            scheduler_output = SimpleNamespace(
                total_num_scheduled_tokens=prompt_ids.numel(),
                num_scheduled_tokens={"runner-request": prompt_ids.numel()},
                scheduled_encoder_inputs={"runner-request": [0, 1]},
            )
            runner_ids, runner_merged, *_ = OmniGPUModelRunner._preprocess(
                runner,
                scheduler_output,
                num_input_tokens=prompt_ids.numel(),
            )
            assert runner_ids is not None
            torch.testing.assert_close(runner_ids, prompt_ids, rtol=0, atol=0)
            assert [item.shape[0] for item in runner_embeddings] == [2, 3]
            torch.testing.assert_close(
                runner_merged[is_multimodal],
                torch.cat(runner_embeddings, dim=0),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                runner_merged[~is_multimodal],
                text_only[~is_multimodal],
                rtol=0,
                atol=0,
            )

            try:
                model._get_audio_embeddings(
                    input_values,
                    padding_mask,
                    counts_cuda + 1,
                    sample=False,
                )
            except ValueError as exc:
                assert "does not match padding_mask" in str(exc)
            else:
                raise AssertionError("Mismatched audio_num_tokens was accepted")

            # Exercise the accepted upper bound on the real Acoustic Encoder.
            max_audio = np.zeros(SAMPLE_RATE * 60, dtype=np.float32)
            max_segment = f"{AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN}"
            max_processed = processor(
                " ".join(max_segment for _ in range(8)),
                mm_items=info.parse_mm_data(
                    {"audio": [(max_audio, SAMPLE_RATE)] * 8}
                ),
            )
            max_mm_data = max_processed["mm_kwargs"].get_data()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            max_embeddings = model._get_audio_embeddings(
                max_mm_data["input_values"].to("cuda"),
                max_mm_data["padding_mask"].to("cuda"),
                max_mm_data["audio_num_tokens"].to("cuda"),
                sample=False,
            )
            assert len(max_embeddings) == 8
            assert all(item.shape == (450, 1536) for item in max_embeddings)
            assert all(torch.isfinite(item).all() for item in max_embeddings)
            torch.cuda.synchronize()
            max_audio_peak_bytes = torch.cuda.max_memory_allocated()

            queue.put(
                {
                    "counts": counts.tolist(),
                    "max_audio_peak_bytes": max_audio_peak_bytes,
                    "max_audio_items": len(max_embeddings),
                    "max_audio_tokens": int(max_embeddings[0].shape[0]),
                    "merged_shape": list(merged.shape),
                    "runner_prefill_verified": True,
                    "sampled_differs_by_seed": True,
                }
            )
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            destroy_model_parallel()
            destroy_distributed_environment()
    except Exception:
        queue.put({"error": traceback.format_exc()})


def test_processor_to_acoustic_prefill_on_gpu():
    model_root = os.getenv("VIBEVOICE_TEST_MODEL_ROOT")
    if not model_root:
        pytest.skip("Set VIBEVOICE_TEST_MODEL_ROOT to run VibeVoice GPU tests")

    model_root_path = Path(model_root)
    model_path = model_root_path / "VibeVoice"
    tokenizer_path = model_root_path / "VibeVoice-1.5B-hf"
    if not (model_path / "model.safetensors.index.json").is_file():
        pytest.fail(f"Official VibeVoice checkpoint not found at {model_path}")
    if not (tokenizer_path / "tokenizer.json").is_file():
        pytest.skip(f"VibeVoice tokenizer fixture not found at {tokenizer_path}")

    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_gpu_processing_worker,
        args=(
            str(model_path),
            str(tokenizer_path),
            _free_port(),
            queue,
        ),
    )
    process.start()
    process.join(timeout=300)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("VibeVoice GPU processing subprocess timed out")

    try:
        result = queue.get(timeout=5)
    except Empty:
        pytest.fail(
            f"GPU subprocess exited with code {process.exitcode} without a result"
        )
    assert "error" not in result, result.get("error")
    assert process.exitcode == 0
    assert result["counts"] == [2, 3]
    assert result["max_audio_items"] == 8
    assert result["max_audio_tokens"] == 450
    assert result["max_audio_peak_bytes"] > 0
    assert result["merged_shape"][1] == 1536
    assert result["runner_prefill_verified"] is True
    assert result["sampled_differs_by_seed"] is True
