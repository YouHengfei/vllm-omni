# SPDX-License-Identifier: Apache-2.0
"""Safety contract for OmniRunner residual-process cleanup."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.helpers import runtime as runtime_helpers
from tests.helpers.runtime import OmniRunner


def test_omni_runner_only_snapshots_owned_engine_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SimpleNamespace(
        pid=101,
        cmdline=lambda: ["vllm::EngineCore"],
        name=lambda: "python",
    )
    unrelated = SimpleNamespace(
        pid=202,
        cmdline=lambda: ["python", "unrelated.py"],
        name=lambda: "python",
    )
    root = SimpleNamespace(children=lambda recursive: [engine, unrelated])
    monkeypatch.setattr(runtime_helpers.psutil, "Process", lambda _pid: root)

    runner = object.__new__(OmniRunner)

    assert runner._owned_engine_processes() == [engine]
