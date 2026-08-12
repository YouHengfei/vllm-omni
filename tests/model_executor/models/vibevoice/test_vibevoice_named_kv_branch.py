# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the fixed-pool named causal KV capability."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.vibevoice.runtime_config import (
    VibeVoiceRuntimeConfig,
)
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
from vllm_omni.worker.named_kv_branch import (
    NamedKVBranchRequest,
    _FixedBlockAllocator,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_named_kv_branch_request_validates_deployment_contract() -> None:
    request = NamedKVBranchRequest(
        name="negative",
        memory_bytes=1024,
        layer_group=0,
        activation_margin_bytes=256,
    )
    assert request.name == "negative"
    assert request.memory_bytes == 1024

    with pytest.raises(ValueError, match="name must be non-empty"):
        NamedKVBranchRequest(name="", memory_bytes=1)
    with pytest.raises(ValueError, match="memory_bytes must be positive"):
        NamedKVBranchRequest(name="negative", memory_bytes=0)
    with pytest.raises(ValueError, match="layer_group must be non-negative"):
        NamedKVBranchRequest(name="negative", memory_bytes=1, layer_group=-1)
    with pytest.raises(ValueError, match="must be non-negative"):
        NamedKVBranchRequest(
            name="negative",
            memory_bytes=1,
            activation_margin_bytes=-1,
        )


def test_fixed_block_allocator_is_deterministic_and_rejects_corruption() -> None:
    allocator = _FixedBlockAllocator(3)
    assert [allocator.allocate(), allocator.allocate(), allocator.allocate()] == [
        0,
        1,
        2,
    ]
    assert allocator.num_free_blocks == 0
    with pytest.raises(RuntimeError, match="exhausted its fixed GPU block pool"):
        allocator.allocate()

    allocator.free([0, 1, 2])
    assert allocator.num_free_blocks == 3
    assert [allocator.allocate(), allocator.allocate(), allocator.allocate()] == [
        0,
        1,
        2,
    ]
    with pytest.raises(ValueError, match="Cannot free unallocated"):
        allocator.free([99])


def test_runtime_config_uses_additional_config_without_touching_hf_config() -> None:
    default = VibeVoiceRuntimeConfig.from_vllm_config(
        SimpleNamespace(additional_config={})
    )
    assert default.negative_kv_cache_memory_bytes == 2 * 1024**3
    assert default.negative_kv_activation_margin_bytes == 512 * 1024**2

    config = SimpleNamespace(
        additional_config={
            "vibevoice_runtime_config": {
                "negative_kv_cache_memory_bytes": "4096",
                "negative_kv_activation_margin_bytes": 128,
                "future_key": True,
            }
        },
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(vibevoice_runtime_config={"wrong": 1})
        ),
    )
    parsed = VibeVoiceRuntimeConfig.from_vllm_config(config)
    assert parsed == VibeVoiceRuntimeConfig(
        negative_kv_cache_memory_bytes=4096,
        negative_kv_activation_margin_bytes=128,
    )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("negative_kv_cache_memory_bytes", 0, "must be positive"),
        (
            "negative_kv_activation_margin_bytes",
            -1,
            "must be non-negative",
        ),
        (
            "negative_kv_cache_memory_bytes",
            True,
            "must be an integer, not bool",
        ),
        (
            "negative_kv_cache_memory_bytes",
            "not-an-int",
            "must be an integer",
        ),
    ],
)
def test_runtime_config_rejects_invalid_capacity_values(
    key: str,
    value: object,
    message: str,
) -> None:
    config = SimpleNamespace(
        additional_config={"vibevoice_runtime_config": {key: value}}
    )
    with pytest.raises(ValueError, match=message):
        VibeVoiceRuntimeConfig.from_vllm_config(config)


def test_undeclared_model_keeps_named_kv_runner_path_disabled() -> None:
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = object()
    runner.named_kv_branches = {}
    OmniGPUModelRunner._maybe_bind_named_kv_branch(runner)
    assert runner.named_kv_branches == {}


def test_runner_rejects_invalid_named_branch_declaration() -> None:
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(named_kv_branch_request={"name": "negative"})
    runner.named_kv_branches = {}
    with pytest.raises(TypeError, match="must be a NamedKVBranchRequest"):
        OmniGPUModelRunner._maybe_bind_named_kv_branch(runner)
    assert runner.named_kv_branches == {}
