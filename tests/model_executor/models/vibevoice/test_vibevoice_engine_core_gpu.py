# SPDX-License-Identifier: Apache-2.0
"""Opt-in real EngineCore coverage for VibeVoice multimodal caching."""

from __future__ import annotations

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


def test_real_engine_core_encoder_cache_and_sampling_lifecycle() -> None:
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
