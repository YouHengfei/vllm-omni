# SPDX-License-Identifier: Apache-2.0
"""Test-only worker snapshot for non-VibeVoice AR TTS isolation gates."""

from __future__ import annotations

from typing import Any


def assert_non_vibevoice_ar_isolation(rpc_result: list[Any]) -> None:
    """Validate the nested stage/rank result returned by collective RPC."""
    assert len(rpc_result) == 1
    rank_results = rpc_result[0]
    assert rank_results
    for snapshot in rank_results:
        assert snapshot["terminal_sample_drain_token_ids"] == []
        assert snapshot["has_terminal_drain_method"] is False
        assert snapshot["postprocess_requires_all_scheduled_requests"] is False
        assert snapshot["named_kv_branch_names"] == []
        assert snapshot["runner_request_ids"] == []
        assert snapshot["model_intermediate_request_ids"] == []


class ARTTSIsolationWorkerExtensionForTest:
    """Expose capability state without changing production model execution."""

    def ar_tts_test_runtime_isolation(self) -> dict[str, Any]:
        runner = self.model_runner
        model = runner.get_model()
        named_branches = getattr(runner, "named_kv_branches", None)
        terminal_token_ids = getattr(model, "terminal_sample_drain_token_ids", None)
        return {
            "rank": int(self.rank),
            "model_type": type(model).__name__,
            "terminal_sample_drain_token_ids": (list(terminal_token_ids) if terminal_token_ids else []),
            "has_terminal_drain_method": callable(getattr(model, "drain_terminal_sampled_tokens", None)),
            "postprocess_requires_all_scheduled_requests": bool(
                getattr(model, "postprocess_requires_all_scheduled_requests", False)
            ),
            "named_kv_branch_names": sorted(named_branches or {}),
            "runner_request_ids": sorted(runner.requests),
            "model_intermediate_request_ids": sorted(runner.model_intermediate_buffer),
        }
