# SPDX-License-Identifier: Apache-2.0
"""Opt-in TP=2 loading and replicated-side-module coverage for VibeVoice."""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tp2_worker(
    rank: int,
    model_path_str: str,
    port: int,
    queue: Any,
) -> None:
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            LOCAL_RANK=str(rank),
            WORLD_SIZE="2",
        )

        import gc
        from math import ceil
        from types import SimpleNamespace

        import torch.distributed as dist
        from vllm.config import (
            get_layers_from_vllm_config,
            set_current_vllm_config,
        )
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.model_executor.layers.attention import Attention
        from vllm.model_executor.model_loader import get_model_loader

        from vllm_omni.engine.arg_utils import OmniEngineArgs
        from vllm_omni.worker.named_kv_branch import (
            NamedCausalKVBranch,
            NamedKVBranchRequest,
        )

        torch.cuda.set_device(rank)
        args = OmniEngineArgs(
            model=model_path_str,
            model_arch="VibeVoiceForConditionalGeneration",
            model_stage="vibevoice",
            worker_type="ar",
            skip_tokenizer_init=True,
            skip_mm_profiling=True,
            dtype="bfloat16",
            load_format="safetensors",
            trust_remote_code=False,
            max_model_len=1024,
            max_num_seqs=1,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            enforce_eager=True,
            enable_prefix_caching=False,
        )
        config = args.create_engine_config()
        init_distributed_environment(
            world_size=2,
            rank=rank,
            local_rank=rank,
            backend="nccl",
        )
        model = None
        try:
            with set_current_vllm_config(config):
                initialize_model_parallel(
                    tensor_model_parallel_size=2,
                    pipeline_model_parallel_size=1,
                )
                model = get_model_loader(config.load_config).load_model(
                    vllm_config=config,
                    model_config=config.model_config,
                )

            params = dict(model.named_parameters())
            projector_weight = params[
                "model.multi_modal_projector.linear_1.weight"
            ]
            semantic_weight = params["model.semantic_connector.linear_1.weight"]
            qkv_weight = params[
                "model.language_model.layers.0.self_attn.qkv_proj.weight"
            ]
            gate_up_weight = params[
                "model.language_model.layers.0.mlp.gate_up_proj.weight"
            ]

            # Transformers acoustic/semantic/projector/diffusion modules are
            # intentionally replicated; only the vLLM Qwen2 backbone is TP
            # sharded.
            assert tuple(projector_weight.shape) == (1536, 64)
            assert tuple(semantic_weight.shape) == (1536, 128)
            assert tuple(qkv_weight.shape) == (1024, 1536)
            assert tuple(gate_up_weight.shape) == (8960, 1536)

            latent_scaling = model.model.latent_scaling_factor.detach().float()
            latent_bias = model.model.latent_bias_factor.detach().float()
            replicated_digest = torch.stack(
                [
                    projector_weight.float().sum(),
                    semantic_weight.float().sum(),
                    latent_scaling.sum(),
                    latent_bias.sum(),
                ]
            )
            gathered_digests = [torch.empty_like(replicated_digest) for _ in range(2)]
            dist.all_gather(gathered_digests, replicated_digest)
            torch.testing.assert_close(gathered_digests[0], gathered_digests[1])

            with torch.inference_mode():
                acoustic_input = (
                    torch.arange(2 * 3 * 64, device="cuda", dtype=torch.float32)
                    .reshape(2, 3, 64)
                    .to(torch.bfloat16)
                    / 100
                )
                projector_output = model.model.multi_modal_projector(acoustic_input)
                diffusion_output = model.model.diffusion_head(
                    torch.zeros(2, 64, device="cuda", dtype=torch.bfloat16),
                    torch.ones(2, device="cuda", dtype=torch.bfloat16),
                    torch.zeros(2, 1536, device="cuda", dtype=torch.bfloat16),
                )

            gathered_projector = [
                torch.empty_like(projector_output) for _ in range(2)
            ]
            gathered_diffusion = [
                torch.empty_like(diffusion_output) for _ in range(2)
            ]
            dist.all_gather(gathered_projector, projector_output)
            dist.all_gather(gathered_diffusion, diffusion_output)
            torch.testing.assert_close(gathered_projector[0], gathered_projector[1])
            torch.testing.assert_close(gathered_diffusion[0], gathered_diffusion[1])
            assert torch.isfinite(projector_output).all()
            assert torch.isfinite(diffusion_output).all()

            # The Qwen shard should not be accidentally replicated. Distinct
            # rank-local sums are sufficient here because shape already proves
            # the expected per-rank partition.
            qwen_digest = qkv_weight.float().sum().reshape(1)
            gathered_qwen = [torch.empty_like(qwen_digest) for _ in range(2)]
            dist.all_gather(gathered_qwen, qwen_digest)
            assert gathered_qwen[0].item() != gathered_qwen[1].item()

            # PR-2: each TP rank owns its sharded negative KV tensor while the
            # shared Qwen forward produces identical full hidden rows.
            attention_layers = get_layers_from_vllm_config(config, Attention)
            layer_names = list(attention_layers)
            first_attention = attention_layers[layer_names[0]]
            kv_spec = first_attention.get_kv_cache_spec(config)
            assert kv_spec is not None
            required_blocks = ceil(
                config.model_config.max_model_len / kv_spec.block_size
            )
            fake_runner = SimpleNamespace(
                vllm_config=config,
                device=torch.device("cuda", rank),
                kv_cache_config=SimpleNamespace(
                    kv_cache_groups=[SimpleNamespace(kv_cache_spec=kv_spec)],
                    num_blocks=required_blocks,
                ),
                attn_groups=[
                    [
                        SimpleNamespace(
                            backend=first_attention.get_attn_backend(),
                            layer_names=layer_names,
                        )
                    ]
                ],
                _kernel_block_sizes=[kv_spec.block_size],
            )
            negative_store = NamedCausalKVBranch(
                runner=fake_runner,
                request=NamedKVBranchRequest(
                    name="negative",
                    memory_bytes=(
                        required_blocks
                        * len(layer_names)
                        * kv_spec.page_size_bytes
                    ),
                ),
            )
            model.bind_named_kv_branch(negative_store)
            negative_request_id = "tp2-negative"
            model._stateful.start_audio_segment(negative_request_id)
            bos_embedding = model.embed_input_ids(
                torch.tensor(
                    [model._stateful.audio_bos_token_id],
                    dtype=torch.long,
                    device="cuda",
                )
            )
            model._stateful.record_positive_condition(
                negative_request_id,
                torch.linspace(
                    -0.5,
                    0.5,
                    1536,
                    dtype=torch.bfloat16,
                    device="cuda",
                ).reshape(1, -1),
            )
            with torch.inference_mode():
                negative_hidden = model._negative_kv_branch.forward_step(
                    [negative_request_id],
                    [bos_embedding],
                )[0]
                model._stateful.record_negative_condition(
                    negative_request_id,
                    negative_hidden,
                )
                torch.manual_seed(12_345)
                next_embeddings, audio_chunks = (
                    model._stateful.process_audio_tokens_batch(
                        request_ids=[negative_request_id],
                        token_embeddings=[bos_embedding],
                        kernel=model.model,
                    )
                )
            gathered_negative = [
                torch.empty_like(negative_hidden) for _ in range(2)
            ]
            dist.all_gather(gathered_negative, negative_hidden)
            torch.testing.assert_close(
                gathered_negative[0],
                gathered_negative[1],
            )
            gathered_next_embedding = [
                torch.empty_like(next_embeddings[0]) for _ in range(2)
            ]
            gathered_audio = [
                torch.empty_like(audio_chunks[0]) for _ in range(2)
            ]
            dist.all_gather(gathered_next_embedding, next_embeddings[0])
            dist.all_gather(gathered_audio, audio_chunks[0])
            torch.testing.assert_close(
                gathered_next_embedding[0],
                gathered_next_embedding[1],
            )
            torch.testing.assert_close(
                gathered_audio[0],
                gathered_audio[1],
            )
            state = model._stateful.get(negative_request_id)
            assert state is not None
            assert state.audio_token_count == 1
            assert len(state.waveform_chunks_cpu) == 1
            assert state.waveform_chunks_cpu[0].shape == (3_200,)
            assert torch.isfinite(state.waveform_chunks_cpu[0]).all()
            assert negative_store.get_sequence_length(negative_request_id) == 1
            model._stateful.cleanup_request(negative_request_id)
            assert negative_store.num_free_blocks == negative_store.num_blocks
            negative_store.close()

            queue.put(
                {
                    "rank": rank,
                    "parameter_bytes": sum(
                        parameter.numel() * parameter.element_size()
                        for parameter in params.values()
                    ),
                    "max_allocated_bytes": torch.cuda.max_memory_allocated(rank),
                }
            )
            dist.barrier()
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            destroy_model_parallel()
            destroy_distributed_environment()
    except Exception:
        queue.put({"rank": rank, "error": traceback.format_exc()})


def test_tp2_replicates_side_modules_and_shards_qwen() -> None:
    model_root = os.getenv("VIBEVOICE_TEST_MODEL_ROOT")
    if not model_root:
        pytest.skip("Set VIBEVOICE_TEST_MODEL_ROOT to run TP=2 coverage")

    model_path = Path(model_root) / "VibeVoice"
    if not (model_path / "model.safetensors.index.json").is_file():
        pytest.fail(f"Official VibeVoice checkpoint not found at {model_path}")

    context = mp.get_context("spawn")
    queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_tp2_worker,
            args=(rank, str(model_path), port, queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=360)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
            pytest.fail("VibeVoice TP=2 subprocess timed out")

    results = []
    try:
        for _ in processes:
            results.append(queue.get(timeout=10))
    except Empty:
        pytest.fail("TP=2 subprocesses exited without returning every rank result")

    errors = [result for result in results if "error" in result]
    assert not errors, errors[0]["error"] if errors else None
    assert all(process.exitcode == 0 for process in processes)
    assert sorted(result["rank"] for result in results) == [0, 1]
    assert results[0]["parameter_bytes"] == results[1]["parameter_bytes"]
    assert all(result["max_allocated_bytes"] > 0 for result in results)
