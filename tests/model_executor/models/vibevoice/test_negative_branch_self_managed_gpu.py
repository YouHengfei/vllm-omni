# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Numeric conformance for the self-managed VibeVoice negative KV branch.

Validates :class:`VibeVoiceSelfManagedNegativeKVStore` — the Plan-B downgrade
that owns the negative Qwen KV in the model instead of the runner — against an
independent HuggingFace Transformers ``Qwen2`` cached reference.

The harness mirrors the upstream negative-KV conformance test: a tiny Qwen2
checkpoint is built once, loaded through vLLM so the shared backbone weights are
identical, then the negative branch is advanced token-by-token while every
returned hidden row is compared against the HF ``past_key_values`` reference.

Runs in a subprocess so the global distributed/model-parallel state it installs
does not leak into the rest of the test session. Requires a GPU.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
import traceback
from queue import Empty
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.gpu]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _self_managed_conformance_worker(port: int, queue: Any) -> None:
    distributed_initialized = False
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        )

        import tempfile

        from transformers import Qwen2Config, Qwen2ForCausalLM
        from vllm.config import set_current_vllm_config
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.model_executor.model_loader import get_model_loader

        from vllm_omni.engine.arg_utils import OmniEngineArgs
        from vllm_omni.model_executor.models.vibevoice.negative_branch import (
            VibeVoiceSelfManagedNegativeKVStore,
        )
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.set_device(0)
        torch.manual_seed(1234)
        hf_config = Qwen2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=28,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_dropout=0.0,
            tie_word_embeddings=False,
            use_cache=True,
        )
        hf_model = Qwen2ForCausalLM(hf_config).eval()

        with tempfile.TemporaryDirectory() as model_dir:
            hf_model.save_pretrained(model_dir, safe_serialization=True)
            args = OmniEngineArgs(
                model=model_dir,
                model_arch="Qwen2ForCausalLM",
                worker_type="ar",
                skip_tokenizer_init=True,
                dtype="bfloat16",
                load_format="safetensors",
                trust_remote_code=False,
                max_model_len=64,
                max_num_seqs=1,
                block_size=16,
                enforce_eager=True,
                enable_prefix_caching=False,
            )
            config = args.create_engine_config()
            init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="nccl")
            distributed_initialized = True
            with set_current_vllm_config(config):
                initialize_model_parallel(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                )
                vllm_model = get_model_loader(config.load_config).load_model(
                    vllm_config=config,
                    model_config=config.model_config,
                )

            hf_model = hf_model.to(device="cuda", dtype=torch.bfloat16)
            language_model = vllm_model.model

            def make_embedding() -> torch.Tensor:
                return torch.randn(
                    1,
                    1,
                    hf_config.hidden_size,
                    device="cuda",
                    dtype=torch.bfloat16,
                )

            def hf_step(embedding: torch.Tensor, past: Any, position: int) -> tuple[torch.Tensor, Any]:
                output = hf_model.model(
                    inputs_embeds=embedding,
                    past_key_values=past,
                    use_cache=True,
                    position_ids=torch.tensor([[position]], device="cuda"),
                    cache_position=torch.tensor([position], device="cuda"),
                    return_dict=True,
                )
                return output.last_hidden_state.reshape(1, -1), output.past_key_values

            store = VibeVoiceSelfManagedNegativeKVStore(
                language_model=language_model,
                hidden_size=hf_config.hidden_size,
            )

            # ---- Single request: 17 tokens (crosses the 16-token page size). ----
            single_embeddings = [make_embedding() for _ in range(17)]
            store.reset_audio_segment("request")
            hf_past = None
            max_abs_diff = 0.0
            with torch.inference_mode():
                for step, embedding in enumerate(single_embeddings):
                    hf_hidden, hf_past = hf_step(embedding, hf_past, step)
                    (store_hidden,) = store.forward_step(["request"], [embedding.reshape(1, -1)])
                    max_abs_diff = max(max_abs_diff, float((store_hidden.float() - hf_hidden.float()).abs().max()))
                    torch.testing.assert_close(store_hidden.float(), hf_hidden.float(), rtol=0.04, atol=0.04)
            store.free("request")

            # ---- Batched staggered requests share one logical step. ----
            # batch-a is 4 tokens ahead of batch-b; every row is compared to its
            # own independent HF cached reference.
            embeddings_a = single_embeddings[:4] + [make_embedding() for _ in range(13)]
            embeddings_b = [make_embedding() for _ in range(13)]
            store.reset_audio_segment("batch-a")
            store.reset_audio_segment("batch-b")
            hf_past_a = None
            hf_past_b = None
            batch_max_abs_diff = 0.0
            with torch.inference_mode():
                # Warm batch-a alone for its first 4 tokens (staggered start).
                for step in range(4):
                    hf_hidden_a, hf_past_a = hf_step(embeddings_a[step], hf_past_a, step)
                    (store_hidden_a,) = store.forward_step(["batch-a"], [embeddings_a[step].reshape(1, -1)])
                    torch.testing.assert_close(store_hidden_a.float(), hf_hidden_a.float(), rtol=0.04, atol=0.04)
                # Advance both in one batched forward_step.
                for index in range(13):
                    step_a = 4 + index
                    step_b = index
                    hf_hidden_a, hf_past_a = hf_step(embeddings_a[step_a], hf_past_a, step_a)
                    hf_hidden_b, hf_past_b = hf_step(embeddings_b[step_b], hf_past_b, step_b)
                    hidden_a, hidden_b = store.forward_step(
                        ["batch-a", "batch-b"],
                        [embeddings_a[step_a].reshape(1, -1), embeddings_b[step_b].reshape(1, -1)],
                    )
                    batch_max_abs_diff = max(
                        batch_max_abs_diff,
                        float((hidden_a.float() - hf_hidden_a.float()).abs().max()),
                        float((hidden_b.float() - hf_hidden_b.float()).abs().max()),
                    )
                    torch.testing.assert_close(hidden_a.float(), hf_hidden_a.float(), rtol=0.04, atol=0.04)
                    torch.testing.assert_close(hidden_b.float(), hf_hidden_b.float(), rtol=0.04, atol=0.04)

            # ---- reset() restarts a request from position 0. ----
            store.reset_audio_segment("batch-a")
            hf_past_a = None
            with torch.inference_mode():
                for step in range(3):
                    hf_hidden_a, hf_past_a = hf_step(embeddings_a[step], hf_past_a, step)
                    (store_hidden_a,) = store.forward_step(["batch-a"], [embeddings_a[step].reshape(1, -1)])
                    torch.testing.assert_close(store_hidden_a.float(), hf_hidden_a.float(), rtol=0.04, atol=0.04)

            store.free("batch-a")
            store.free("batch-b")
            queue.put(
                {
                    "ok": True,
                    "max_abs_diff": max_abs_diff,
                    "batch_max_abs_diff": batch_max_abs_diff,
                }
            )
    except Exception:
        queue.put({"ok": False, "traceback": traceback.format_exc()})
    finally:
        if distributed_initialized:
            try:
                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception:
                pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_self_managed_negative_kv_conformance() -> None:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_self_managed_conformance_worker,
        args=(_free_port(), queue),
    )
    process.start()
    process.join(timeout=600)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("self-managed negative-KV conformance worker timed out")
    try:
        result = queue.get_nowait()
    except Empty:
        pytest.fail(f"worker exited without a result (exitcode={process.exitcode})")
    if not result.get("ok"):
        pytest.fail(result.get("traceback", "unknown worker failure"))
