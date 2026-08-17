# SPDX-License-Identifier: Apache-2.0
"""Phase A perf harness contracts: disabled by default, opt-in accumulation."""

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.vibevoice import perf_timing

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _restore_timing_state():
    previous = perf_timing.level()
    perf_timing.set_level(0)
    perf_timing.reset()
    yield
    perf_timing.set_level(previous)
    perf_timing.reset()


def test_disabled_by_default_and_record_is_noop() -> None:
    assert perf_timing.level() == 0
    with perf_timing.record("positive_forward"):
        pass
    assert perf_timing.snapshot() == {}


def test_level_one_accumulates_counts_and_totals() -> None:
    perf_timing.set_level(1)
    with perf_timing.record("diffusion"):
        pass
    with perf_timing.record("diffusion"):
        pass
    with perf_timing.record("m4a_decode"):
        pass

    snapshot = perf_timing.snapshot()
    assert snapshot["diffusion"][0] == 2
    assert snapshot["m4a_decode"][0] == 1
    assert all(total >= 0.0 for _, total in snapshot.values())


def test_level_clamps_and_reset_clears() -> None:
    perf_timing.set_level(99)
    assert perf_timing.level() == 2
    perf_timing.set_level(-3)
    assert perf_timing.level() == 0
    perf_timing.set_level(1)
    with perf_timing.record("preprocess"):
        pass
    assert perf_timing.snapshot() != {}
    perf_timing.reset()
    assert perf_timing.snapshot() == {}
