# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-gated per-phase timing harness for the VibeVoice decode path.

Phase A of the performance plan (see README_VIBEVOICE.md §13) needs a
repeatable per-token phase breakdown without paying for it in production.
Set ``VLLM_OMNI_VIBEVOICE_PERF_TIMING`` to enable:

- ``1``: CPU wall timers around each phase (enqueue-side cost; safe under
  ``async_scheduling`` because nothing here synchronizes the CUDA stream).
- ``2``: additionally synchronize the stream at every phase boundary so the
  numbers include GPU execution. This distorts pipelining on purpose and is
  only for dedicated benchmark runs.

Accumulators are process-global rolling counters; a summary line is logged
every ``_LOG_EVERY`` positive-forward steps. Disabling (default) costs one
integer comparison per instrumentation site.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_VAR = "VLLM_OMNI_VIBEVOICE_PERF_TIMING"
_LOG_EVERY = 64


def _parse_level() -> int:
    raw = os.environ.get(ENV_VAR, "")
    try:
        level = int(raw) if raw else 0
    except ValueError:
        logger.warning("%s=%r is not an integer; timing disabled.", ENV_VAR, raw)
        return 0
    return max(0, min(level, 2))


_LEVEL = _parse_level()

_counts: dict[str, int] = {}
_totals_s: dict[str, float] = {}


def level() -> int:
    return _LEVEL


def set_level(value: int) -> None:
    """Test hook: override the env-derived level without re-importing."""
    global _LEVEL
    _LEVEL = max(0, min(int(value), 2))


def reset() -> None:
    _counts.clear()
    _totals_s.clear()


@contextmanager
def record(phase: str) -> Iterator[None]:
    """Time one phase occurrence; a no-op when the harness is disabled."""
    if _LEVEL == 0:
        yield
        return
    sync = _LEVEL >= 2
    if sync:
        torch.accelerator.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if sync:
            torch.accelerator.synchronize()
        elapsed = time.perf_counter() - start
        _counts[phase] = _counts.get(phase, 0) + 1
        _totals_s[phase] = _totals_s.get(phase, 0.0) + elapsed
        if phase == "positive_forward" and _counts[phase] % _LOG_EVERY == 0:
            log_summary()


def snapshot() -> dict[str, tuple[int, float]]:
    """Return ``{phase: (count, total_seconds)}`` for tests and diagnostics."""
    return {phase: (_counts[phase], _totals_s[phase]) for phase in _counts}


def log_summary() -> None:
    parts = []
    for phase in sorted(_counts):
        count = _counts[phase]
        if count:
            parts.append(f"{phase}={_totals_s[phase] / count * 1e3:.3f}ms x{count}")
    logger.info("VibeVoice perf timing (level=%d): %s", _LEVEL, ", ".join(parts))


__all__ = ["ENV_VAR", "level", "log_summary", "record", "reset", "set_level", "snapshot"]
