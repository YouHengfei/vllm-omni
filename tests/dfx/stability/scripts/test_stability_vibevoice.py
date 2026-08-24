# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VibeVoice stability: OmniServer + ``vllm bench serve --omni`` for a fixed duration.

Configuration: ``tests/dfx/stability/tests/test_vibevoice.json``.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from tests.dfx.conftest import (
    create_benchmark_indices,
    create_test_parameter_mapping,
    create_unique_server_params,
    load_configs,
)
from tests.dfx.stability.helpers import _run_one_vllm_bench_batch, run_stability_benchmark_loop
from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path

STABILITY_DIR = Path(__file__).resolve().parent.parent
DEPLOY_CONFIGS_DIR = STABILITY_DIR / "deploy"
CONFIG_FILE_PATH = str(STABILITY_DIR / "tests" / "test_vibevoice.json")
DEFAULT_NUM_PROMPTS_PER_BATCH = 20
STABILITY_SERVER_TIMEOUT_ARGS = ["--stage-init-timeout", "600"]
_REFERENCE_AUDIO = get_asset_path("qwen3_tts/clone_2.wav").resolve().as_uri()

# The stability client and server are always local. Do not allow a developer or
# CI HTTP proxy to turn localhost Speech requests into 502 responses.
for proxy_bypass_env in ("NO_PROXY", "no_proxy"):
    current_bypass = os.environ.get(proxy_bypass_env, "")
    entries = [entry for entry in current_bypass.split(",") if entry]
    for local_host in ("127.0.0.1", "localhost"):
        if local_host not in entries:
            entries.append(local_host)
    os.environ[proxy_bypass_env] = ",".join(entries)

try:
    BENCHMARK_CONFIGS = load_configs(CONFIG_FILE_PATH)
except FileNotFoundError:
    BENCHMARK_CONFIGS = []

# Keep the checked-in JSON portable while allowing an exact local checkpoint
# and tokenizer to be selected without encoding a developer filesystem path.
for config in BENCHMARK_CONFIGS:
    server_params = config["server_params"]
    benchmark_params = config["benchmark_params"]
    model = os.getenv("VIBEVOICE_TEST_MODEL")
    tokenizer = os.getenv("VIBEVOICE_TEST_TOKENIZER")
    if model:
        server_params["model"] = model
    serve_args = server_params.setdefault("serve_args", {})
    if tokenizer:
        serve_args["tokenizer"] = tokenizer
    if revision := os.getenv("VIBEVOICE_TEST_MODEL_REVISION"):
        serve_args["revision"] = revision
    if tokenizer_revision := os.getenv("VIBEVOICE_TEST_TOKENIZER_REVISION"):
        serve_args["tokenizer_revision"] = tokenizer_revision
    if deploy_config := os.getenv("VIBEVOICE_TEST_DEPLOY_CONFIG"):
        extra_cli_args = server_params.setdefault("extra_cli_args", [])
        extra_cli_args.extend(["--deploy-config", str(Path(deploy_config).expanduser().resolve())])
    # A WAV data URL can exceed the host's argv limit because the benchmark
    # forwards ``extra_body`` through the CLI. Use a repository-relative file
    # URI and grant access only to the vendored asset directory instead.
    assets_path = get_asset_path("").resolve()
    server_params.setdefault("serve_args", {})["allowed_local_media_path"] = str(assets_path)
    for params in benchmark_params:
        if tokenizer:
            params["tokenizer"] = tokenizer
        params.setdefault("extra_body", {})["ref_audio"] = _REFERENCE_AUDIO

test_params = create_unique_server_params(BENCHMARK_CONFIGS, DEPLOY_CONFIGS_DIR) if BENCHMARK_CONFIGS else []
server_to_benchmark_mapping = create_test_parameter_mapping(BENCHMARK_CONFIGS) if BENCHMARK_CONFIGS else {}
benchmark_indices = create_benchmark_indices(BENCHMARK_CONFIGS, server_to_benchmark_mapping)


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@pytest.mark.parametrize("stability_benchmark_params", benchmark_indices, indirect=True)
def test_stability_vibevoice(omni_server, stability_benchmark_params):
    test_name = stability_benchmark_params["test_name"]
    params = stability_benchmark_params["params"]
    duration_sec = int(os.getenv("VIBEVOICE_STABILITY_DURATION_SEC", params.get("duration_sec", 300)))
    num_prompts_per_batch = int(
        os.getenv(
            "VIBEVOICE_STABILITY_NUM_PROMPTS_PER_BATCH",
            params.get("num_prompts_per_batch", DEFAULT_NUM_PROMPTS_PER_BATCH),
        )
    )
    request_rate = params.get("request_rate")
    max_concurrency = int(os.getenv("VIBEVOICE_STABILITY_MAX_CONCURRENCY", params.get("max_concurrency", 4)))
    if duration_sec < 1:
        raise ValueError("VIBEVOICE_STABILITY_DURATION_SEC must be positive")
    if num_prompts_per_batch < 1:
        raise ValueError("VIBEVOICE_STABILITY_NUM_PROMPTS_PER_BATCH must be positive")
    if not 1 <= max_concurrency <= 4:
        raise ValueError("VIBEVOICE_STABILITY_MAX_CONCURRENCY must be between 1 and 4")

    result_dir = Path(
        os.getenv(
            "VIBEVOICE_STABILITY_RESULT_DIR",
            STABILITY_DIR / "results" / "vibevoice",
        )
    ).expanduser()
    result_dir.mkdir(parents=True, exist_ok=True)

    bench_params = {
        k: v
        for k, v in params.items()
        if k not in ("duration_sec", "request_rate", "max_concurrency", "num_prompts_per_batch")
    }
    if value := os.getenv("VIBEVOICE_STABILITY_RANDOM_INPUT_LEN"):
        bench_params["random_input_len"] = int(value)
    if value := os.getenv("VIBEVOICE_STABILITY_RANDOM_OUTPUT_LEN"):
        bench_params["random_output_len"] = int(value)

    result = run_stability_benchmark_loop(
        host=omni_server.host,
        port=omni_server.port,
        model=omni_server.model,
        duration_sec=duration_sec,
        params=bench_params,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        result_dir=str(result_dir.resolve()),
        num_prompts_per_batch=num_prompts_per_batch,
        run_one_batch=_run_one_vllm_bench_batch,
        result_filename="vibevoice_stability_summary.json",
    )

    with httpx.Client(trust_env=False, timeout=300.0) as client:
        health = client.get(f"http://{omni_server.host}:{omni_server.port}/health")
        probe = client.post(
            f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech",
            json={
                "model": omni_server.model,
                "input": "Final health probe after the stability run.",
                "ref_audio": _REFERENCE_AUDIO,
                "response_format": "pcm",
                "max_new_tokens": 2,
            },
        )

    assert result.get("failed", 0) == 0, f"[{test_name}] Failed requests detected: {result.get('errors', [])}"
    assert result.get("completed", 0) > 0, f"[{test_name}] No requests completed"
    assert health.status_code == 200, health.text
    assert probe.status_code == 200, probe.text
    assert probe.headers.get("X-Finish-Reason") == "length"
    assert len(probe.content) == 2 * 3_200 * 2
    print(f"\n[{test_name}] Stability benchmark completed: {result}")
