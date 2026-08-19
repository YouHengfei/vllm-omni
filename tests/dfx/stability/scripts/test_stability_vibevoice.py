# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VibeVoice stability: OmniServer + ``vllm bench serve --omni`` for a fixed duration.

Configuration: ``tests/dfx/stability/tests/test_vibevoice.json``.

Aligned with VoxCPM2 stability (same conftest + helpers), with VibeVoice-specific
differences: max_concurrency=4 (matches max_num_seqs=4), stream=true (SSE for
continuity/underrun metrics), and ref_audio in extra_body (VibeVoice requires
reference audio for voice cloning).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.dfx.conftest import (
    create_benchmark_indices,
    create_test_parameter_mapping,
    create_unique_server_params,
    load_configs,
)
from tests.dfx.stability.helpers import run_stability_benchmark_loop

STABILITY_DIR = Path(__file__).resolve().parent.parent
DEPLOY_CONFIGS_DIR = STABILITY_DIR / "deploy"
CONFIG_FILE_PATH = str(STABILITY_DIR / "tests" / "test_vibevoice.json")
DEFAULT_NUM_PROMPTS_PER_BATCH = 20
STABILITY_SERVER_TIMEOUT_ARGS = ["--stage-init-timeout", "600"]

try:
    BENCHMARK_CONFIGS = load_configs(CONFIG_FILE_PATH)
except FileNotFoundError:
    BENCHMARK_CONFIGS = []

test_params = create_unique_server_params(BENCHMARK_CONFIGS, DEPLOY_CONFIGS_DIR) if BENCHMARK_CONFIGS else []
server_to_benchmark_mapping = create_test_parameter_mapping(BENCHMARK_CONFIGS) if BENCHMARK_CONFIGS else {}
benchmark_indices = create_benchmark_indices(BENCHMARK_CONFIGS, server_to_benchmark_mapping)


@pytest.mark.slow
@pytest.mark.tts
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@pytest.mark.parametrize("stability_benchmark_params", benchmark_indices, indirect=True)
def test_stability_vibevoice(omni_server, stability_benchmark_params):
    test_name = stability_benchmark_params["test_name"]
    params = stability_benchmark_params["params"]
    duration_sec = params.get("duration_sec", 300)
    num_prompts_per_batch = params.get("num_prompts_per_batch", DEFAULT_NUM_PROMPTS_PER_BATCH)
    request_rate = params.get("request_rate")
    max_concurrency = params.get("max_concurrency")

    bench_params = {
        k: v
        for k, v in params.items()
        if k not in ("duration_sec", "request_rate", "max_concurrency", "num_prompts_per_batch")
    }

    result = run_stability_benchmark_loop(
        host=omni_server.host,
        port=omni_server.port,
        model=omni_server.model,
        duration_sec=duration_sec,
        params=bench_params,
        num_prompts_per_batch=num_prompts_per_batch,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        timeout=omni_server.init_timeout,
    )

    assert result is not None, "Stability benchmark did not produce a result"
    print(f"\n[{test_name}] Stability benchmark completed: {result}")
