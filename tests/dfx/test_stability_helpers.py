# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fail-closed contracts for duration-based stability benchmarks."""

from __future__ import annotations

import pytest

from tests.dfx.stability.helpers import (
    _normalize_bench_metrics,
    classify_speech_batch,
    merge_batch_results,
    run_stability_benchmark_loop,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_empty_fallback_result_is_a_failed_batch() -> None:
    assert _normalize_bench_metrics({"completed": 0, "failed": 0, "duration": 0}) == {
        "completed": 0,
        "failed": 1,
        "duration": 0.0,
        "errors": ["benchmark produced no completed or failed request records"],
    }


def test_speech_finish_reasons_classify_length_as_success() -> None:
    result = classify_speech_batch(
        {
            "completed": 5,
            "failed": 1,
            "duration": 2.0,
            "errors": ["transport failed"],
            "speech_finish_reason_counts": {"stop": 3, "length": 2},
        }
    )

    assert result["successful_requests"] == 5
    assert result["natural_stop_requests"] == 3
    assert result["truncated_requests"] == 2
    assert result["request_failures"] == 1
    assert result["completed"] == 5
    assert result["failed"] == 1


def test_speech_finish_reason_protocol_failures_are_separate() -> None:
    result = classify_speech_batch(
        {
            "completed": 4,
            "failed": 0,
            "duration": 1.0,
            "errors": [],
            "speech_finish_reason_counts": {"stop": 1, "length": 1, "cancelled": 1},
        }
    )

    assert result["successful_requests"] == 2
    assert result["natural_stop_requests"] == 1
    assert result["truncated_requests"] == 1
    assert result["request_failures"] == 2
    assert result["failed"] == 2
    assert len(result["errors"]) == 2


def test_merge_batch_results_preserves_speech_classification() -> None:
    merged = merge_batch_results(
        [
            classify_speech_batch(
                {
                    "completed": 2,
                    "failed": 0,
                    "duration": 1.0,
                    "errors": [],
                    "speech_finish_reason_counts": {"stop": 1, "length": 1},
                }
            ),
            classify_speech_batch(
                {
                    "completed": 1,
                    "failed": 0,
                    "duration": 1.0,
                    "errors": [],
                    "speech_finish_reason_counts": {"stop": 1},
                }
            ),
        ],
        total_duration_sec=2.0,
    )

    assert merged["successful_requests"] == 3
    assert merged["natural_stop_requests"] == 2
    assert merged["truncated_requests"] == 1
    assert merged["request_failures"] == 0


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
