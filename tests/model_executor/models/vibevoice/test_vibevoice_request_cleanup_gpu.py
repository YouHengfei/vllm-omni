# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU contract for aborting with a pending waveform D2H transfer."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.vibevoice.stateful import VibeVoiceRequestState

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def test_request_state_clear_waits_for_pending_waveform_copy() -> None:
    state = VibeVoiceRequestState(
        request_id="request-a",
        guidance_scale=1.3,
        num_diffusion_steps=10,
    )
    source = torch.arange(1_048_576, device="cuda", dtype=torch.float32)
    buffer = torch.empty_like(source, device="cpu", pin_memory=True)
    buffer.copy_(source, non_blocking=True)
    event = torch.cuda.Event()
    event.record()
    state.waveform_chunks_cpu.append(buffer)
    state._waveform_events[id(buffer)] = (event, buffer)

    state.clear()

    assert event.query()
    assert torch.equal(buffer, torch.arange(buffer.numel(), dtype=torch.float32))
    assert state.waveform_chunks_cpu == []
    assert state._waveform_events == {}
    assert state._pinned_pool == []
