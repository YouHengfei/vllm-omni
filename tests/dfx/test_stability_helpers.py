# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fail-closed contracts for duration-based stability benchmarks."""

from __future__ import annotations

import pytest

from tests.dfx.stability.helpers import _normalize_bench_metrics, run_stability_benchmark_loop

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_empty_fallback_result_is_a_failed_batch() -> None:
    assert _normalize_bench_metrics({"completed": 0, "failed": 0, "duration": 0}) == {
        "completed": 0,
        "failed": 1,
        "duration": 0.0,
        "errors": ["benchmark produced no completed or failed request records"],
    }


def test_stability_loop_stops_after_first_failed_batch(tmp_path) -> None:
    calls = 0

    def run_one_batch(*_args):
        nonlocal calls
        calls += 1
        return {
            "completed": 7,
            "failed": 1,
            "duration": 1.0,
            "errors": ["engine died"],
        }

    result = run_stability_benchmark_loop(
        host="127.0.0.1",
        port=8000,
        model="model",
        duration_sec=3600,
        params={},
        request_rate=None,
        max_concurrency=4,
        result_dir=str(tmp_path),
        num_prompts_per_batch=20,
        run_one_batch=run_one_batch,
    )

    assert calls == 1
    assert result["completed"] == 7
    assert result["failed"] == 1
    assert result["errors"] == ["engine died"]
