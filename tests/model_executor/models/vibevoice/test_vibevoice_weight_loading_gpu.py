# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in GPU integration test for the complete official VibeVoice checkpoint."""

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
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _load_checkpoint_tensor(model_path: Path, name: str) -> torch.Tensor:
    import json

    from safetensors import safe_open

    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    shard = model_path / index["weight_map"][name]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def _gpu_load_worker(model_path_str: str, port: int, queue: Any) -> None:
    """Load in a fresh process so vLLM distributed globals cannot leak."""
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        )

        import gc
        from math import ceil
        from types import SimpleNamespace

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

        model_path = Path(model_path_str)
        torch.cuda.set_device(0)
        args = OmniEngineArgs(
            model=str(model_path),
            model_arch="VibeVoiceForConditionalGeneration",
            model_stage="latent_generator",
            worker_type="ar",
            skip_tokenizer_init=True,
            dtype="bfloat16",
            load_format="safetensors",
            trust_remote_code=False,
            max_model_len=4096,
            max_num_seqs=1,
            enforce_eager=True,
            enable_prefix_caching=False,
        )
        config = args.create_engine_config()
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="nccl")
        model = None
        try:
            with set_current_vllm_config(config):
                initialize_model_parallel(
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                )
                loader = get_model_loader(config.load_config)
                model = loader.load_model(
                    vllm_config=config,
                    model_config=config.model_config,
                )

            params = dict(model.named_parameters())
            assert params
            assert {param.dtype for param in params.values()} == {torch.bfloat16}
            assert {param.device.type for param in params.values()} == {"cuda"}

            # Directly mapped modules.
            direct_pairs = {
                "model.multi_modal_projector.linear_1.weight": (
                    "model.acoustic_connector.fc1.weight"
                ),
                "model.audio_tower.encoder.stem.conv.conv.weight": (
                    "model.acoustic_tokenizer.encoder.downsample_layers.0.0.conv.conv.weight"
                ),
                "model.semantic_tokenizer_encoder.head.conv.weight": (
                    "model.semantic_tokenizer.encoder.head.conv.conv.weight"
                ),
                "model.diffusion_head.cond_proj.weight": "model.prediction_head.cond_proj.weight",
                "model.latent_scaling_factor": "model.speech_scaling_factor",
                "model.latent_bias_factor": "model.speech_bias_factor",
            }
            for runtime_name, checkpoint_name in direct_pairs.items():
                expected = _load_checkpoint_tensor(model_path, checkpoint_name)
                actual = params[runtime_name].detach().cpu()
                assert torch.equal(actual, expected), runtime_name

            # Qwen2's child load_weights() must pack source shards in vLLM order.
            qkv_expected = torch.cat(
                [
                    _load_checkpoint_tensor(
                        model_path,
                        f"model.language_model.layers.0.self_attn.{proj}.weight",
                    )
                    for proj in ("q_proj", "k_proj", "v_proj")
                ],
                dim=0,
            )
            qkv_actual = params[
                "model.language_model.layers.0.self_attn.qkv_proj.weight"
            ].detach().cpu()
            assert torch.equal(qkv_actual, qkv_expected)

            gate_up_expected = torch.cat(
                [
                    _load_checkpoint_tensor(
                        model_path,
                        f"model.language_model.layers.0.mlp.{proj}.weight",
                    )
                    for proj in ("gate_proj", "up_proj")
                ],
                dim=0,
            )
            gate_up_actual = params[
                "model.language_model.layers.0.mlp.gate_up_proj.weight"
            ].detach().cpu()
            assert torch.equal(gate_up_actual, gate_up_expected)

            # The loaded side module must be executable in the configured dtype.
            with torch.inference_mode():
                diffusion_output = model.model.diffusion_head(
                    torch.randn(2, 64, device="cuda", dtype=torch.bfloat16),
                    torch.ones(2, device="cuda", dtype=torch.bfloat16),
                    torch.randn(2, 1536, device="cuda", dtype=torch.bfloat16),
                )
            assert diffusion_output.shape == (2, 64)
            assert torch.isfinite(diffusion_output).all()

            # M4a: execute the complete model-local CFG + DPM numerical loop
            # with explicit noise. Request/KV/decoder state is intentionally
            # outside this weight-loading worker.
            with torch.inference_mode():
                positive_condition = (
                    torch.arange(1536, device="cuda", dtype=torch.float32)
                    .reshape(1, 1536)
                    .to(torch.bfloat16)
                    / 1536
                )
                negative_condition = -positive_condition
                diffusion_noise = (
                    torch.arange(2 * 64, device="cuda", dtype=torch.float32)
                    .reshape(2, 64)
                    .to(torch.bfloat16)
                    / 64
                    - 1
                )
                sampled_latent = model.model.sample_audio_latent(
                    positive_condition,
                    negative_condition,
                    diffusion_noise,
                    guidance_scale=1.3,
                    num_inference_steps=10,
                )
            assert sampled_latent.shape == (1, 1, 64)
            assert sampled_latent.dtype == torch.bfloat16
            assert sampled_latent.device.type == "cuda"
            assert torch.isfinite(sampled_latent).all()

            # M4b: two consecutive per-request chunks must thread independent
            # acoustic/semantic causal caches and reproduce full causal decode.
            decode_latent = torch.linspace(
                -0.5,
                0.5,
                64,
                device="cuda",
                dtype=torch.bfloat16,
            ).reshape(1, 1, 64)
            second_latent = decode_latent.flip(-1) * 0.5
            with torch.inference_mode():
                first_chunk = model.model.decode_audio_token(decode_latent)
                second_chunk = model.model.decode_audio_token(
                    second_latent,
                    acoustic_cache=first_chunk.acoustic_cache,
                    semantic_cache=first_chunk.semantic_cache,
                )
            for chunk in (first_chunk, second_chunk):
                assert chunk.audio.shape == (1, 1, 3_200)
                assert chunk.semantic_latent.shape == (1, 1, 128)
                assert chunk.next_embedding.shape == (1, 1, 1536)
                assert chunk.audio.dtype == torch.bfloat16
                assert chunk.next_embedding.dtype == torch.bfloat16
                assert chunk.audio.device.type == "cuda"
                assert torch.isfinite(chunk.audio).all()
                assert torch.isfinite(chunk.semantic_latent).all()
                assert torch.isfinite(chunk.next_embedding).all()
            assert first_chunk.acoustic_cache is second_chunk.acoustic_cache
            assert first_chunk.semantic_cache is second_chunk.semantic_cache
            assert first_chunk.acoustic_cache is not first_chunk.semantic_cache

            combined_latent = torch.cat([decode_latent, second_latent], dim=1)
            decoder_latent = (
                combined_latent
                / model.model.latent_scaling_factor.to(combined_latent)
                - model.model.latent_bias_factor.to(combined_latent)
            )
            with torch.inference_mode():
                full_audio = model.model.audio_tower.decode(
                    decoder_latent,
                    use_cache=False,
                ).audio
                full_semantic = model.model.semantic_tokenizer_encoder(
                    full_audio,
                    use_cache=False,
                ).latents
            cached_audio = torch.cat(
                [first_chunk.audio, second_chunk.audio],
                dim=-1,
            )
            cached_semantic = torch.cat(
                [first_chunk.semantic_latent, second_chunk.semantic_latent],
                dim=1,
            )
            torch.testing.assert_close(cached_audio, full_audio)
            # BF16 convolution kernels are shape-dependent: streaming one
            # 3200-sample chunk at a time is not bit-exact with one 6400-sample
            # call even though the causal cache is correct. The observed
            # semantic max abs drift on H100 is 0.125; keep a bounded parity
            # guard while treating cached chunking as the official path.
            assert (cached_semantic - full_semantic).abs().max().item() <= 0.25

            # PR-2: bind the production VibeVoice negative executor to a
            # minimal fixed store and advance official Qwen weights twice.
            attention_layers = get_layers_from_vllm_config(config, Attention)
            layer_names = list(attention_layers)
            assert len(layer_names) == 28
            first_attention = attention_layers[layer_names[0]]
            kv_spec = first_attention.get_kv_cache_spec(config)
            assert kv_spec is not None
            backend = first_attention.get_attn_backend()
            required_blocks = ceil(
                config.model_config.max_model_len / kv_spec.block_size
            )
            fake_runner = SimpleNamespace(
                vllm_config=config,
                device=torch.device("cuda"),
                kv_cache_config=SimpleNamespace(
                    kv_cache_groups=[SimpleNamespace(kv_cache_spec=kv_spec)],
                    num_blocks=required_blocks,
                ),
                attn_groups=[
                    [
                        SimpleNamespace(
                            backend=backend,
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
            request_id = "official-negative"
            model._stateful.start_audio_segment(request_id)
            bos_embedding = model.embed_input_ids(
                torch.tensor(
                    [model._stateful.audio_bos_token_id],
                    dtype=torch.long,
                    device="cuda",
                )
            )
            feedback_embedding = torch.linspace(
                -0.25,
                0.25,
                1536,
                dtype=torch.bfloat16,
                device="cuda",
            ).reshape(1, -1)
            with torch.inference_mode():
                first_negative = model._negative_kv_branch.forward_step(
                    [request_id],
                    [bos_embedding],
                )[0]
                second_negative = model._negative_kv_branch.forward_step(
                    [request_id],
                    [feedback_embedding],
                )[0]
            assert first_negative.shape == (1, 1536)
            assert second_negative.shape == (1, 1536)
            assert first_negative.dtype == torch.bfloat16
            assert torch.isfinite(first_negative).all()
            assert torch.isfinite(second_negative).all()
            assert not torch.equal(first_negative, second_negative)
            assert negative_store.get_sequence_length(request_id) == 2
            model._stateful.cleanup_request(request_id)
            assert negative_store.num_free_blocks == negative_store.num_blocks
            negative_store.close()

            queue.put(
                {
                    "class_name": type(model).__name__,
                    "num_parameters": len(params),
                    "parameter_bytes": sum(
                        param.numel() * param.element_size() for param in params.values()
                    ),
                    "negative_layers": len(layer_names),
                    "negative_steps": 2,
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


def test_complete_official_checkpoint_loads_on_gpu():
    model_root = os.getenv("VIBEVOICE_TEST_MODEL_ROOT")
    if not model_root:
        pytest.skip("Set VIBEVOICE_TEST_MODEL_ROOT to run full GPU weight loading")

    model_path = Path(model_root) / "VibeVoice"
    if not (model_path / "model.safetensors.index.json").is_file():
        pytest.fail(f"Official VibeVoice checkpoint not found at {model_path}")

    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_gpu_load_worker,
        args=(str(model_path), _free_port(), queue),
    )
    process.start()
    process.join(timeout=300)
    if process.is_alive():
        process.kill()
        process.join()
        pytest.fail("VibeVoice GPU weight-loading subprocess timed out")

    try:
        result = queue.get(timeout=5)
    except Empty:
        pytest.fail(f"GPU subprocess exited with code {process.exitcode} without a result")
    assert "error" not in result, result.get("error")
    assert process.exitcode == 0
    assert result == {
        "class_name": "VibeVoiceForConditionalGeneration",
        "num_parameters": 1064,
        "parameter_bytes": 5_408_043_974,
        "negative_layers": 28,
        "negative_steps": 2,
    }
