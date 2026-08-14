# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the non-VibeVoice AR isolation worker snapshot."""

from types import SimpleNamespace

from tests.helpers.ar_tts_isolation_worker_extension import (
    ARTTSIsolationWorkerExtensionForTest,
    assert_non_vibevoice_ar_isolation,
)


def test_non_vibevoice_ar_isolation_snapshot_and_assertion() -> None:
    model = SimpleNamespace()
    runner = SimpleNamespace(
        get_model=lambda: model,
        named_kv_branches={},
        requests={},
        model_intermediate_buffer={},
    )
    worker = SimpleNamespace(model_runner=runner, rank=0)

    snapshot = ARTTSIsolationWorkerExtensionForTest.ar_tts_test_runtime_isolation(worker)

    assert snapshot["model_type"] == "SimpleNamespace"
    assert_non_vibevoice_ar_isolation([[snapshot]])
