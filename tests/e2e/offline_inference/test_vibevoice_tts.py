# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E test for VibeVoice native AR offline inference."""

import os
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from tests.helpers.ar_tts_isolation_worker_extension import (
    assert_non_vibevoice_ar_isolation,
)
from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path

_MODEL_ROOT = os.getenv("VIBEVOICE_TEST_MODEL_ROOT")
_MODEL = str(Path(_MODEL_ROOT) / "VibeVoice") if _MODEL_ROOT else "VibeVoice"
_DEPLOY_CONFIG = get_deploy_config_path("vibevoice.yaml")
_SAMPLE_RATE = 24_000

# VibeVoice uses a shared AR runner; the isolation extension asserts no
# VibeVoice-specific capability leaks into a non-VibeVoice AR model's state.
_WORKER_EXTENSION = "tests.helpers.ar_tts_isolation_worker_extension.ARTTSIsolationWorkerExtensionForTest"
_OMNI_RUNNER_PARAM = (
    _MODEL,
    _DEPLOY_CONFIG,
    {
        "trust_remote_code": False,
        "stage_overrides": {"0": {"worker_extension_cls": _WORKER_EXTENSION}},
    },
)

pytestmark = pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True)


def _extract_audio(multimodal_output: dict) -> torch.Tensor:
    """Extract the final complete audio tensor from multimodal output."""
    assert isinstance(multimodal_output, (dict, Mapping)), f"Expected dict/Mapping, got {type(multimodal_output)}"

    audio = multimodal_output.get("audio")
    if audio is None:
        audio = multimodal_output.get("model_outputs")
    assert audio is not None, f"No audio key, got {list(multimodal_output.keys())}"

    if isinstance(audio, list):
        valid = [torch.as_tensor(x).float().cpu().reshape(-1) for x in audio if x is not None]
        assert valid, "No valid audio tensors in output list"
        audio = torch.cat(valid, dim=0) if len(valid) > 1 else valid[0]

    assert isinstance(audio, torch.Tensor), f"Expected Tensor, got {type(audio)}"
    return audio


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_vibevoice_zero_shot_001(omni_runner: OmniRunner) -> None:
    """Test zero-shot TTS produces valid audio output."""
    outputs = omni_runner.omni.generate([{"prompt": "Hello, this is a test."}])
    assert len(outputs) == 1

    audio = _extract_audio(outputs[0].outputs[0].multimodal_output)
    duration_s = audio.shape[0] / _SAMPLE_RATE
    assert 0.5 < duration_s < 30.0, f"Audio duration out of range: {duration_s:.2f}s"
    assert audio.shape[0] % 3_200 == 0, f"Audio length not a multiple of 3200: {audio.shape[0]}"
    assert torch.isfinite(audio).all(), "Audio contains non-finite values"


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_vibevoice_voice_clone_002(omni_runner: OmniRunner) -> None:
    """Test voice cloning with a vendored reference audio file."""
    ref_path = str(get_asset_path("qwen3_tts/clone_2.wav"))

    outputs = omni_runner.omni.generate(
        [
            {
                "prompt": "Hello, this is a voice clone demo.",
                "additional_information": {"reference_audio": ref_path},
            }
        ]
    )
    assert len(outputs) == 1

    audio = _extract_audio(outputs[0].outputs[0].multimodal_output)
    duration_s = audio.shape[0] / _SAMPLE_RATE
    assert 0.5 < duration_s < 30.0, f"Audio duration out of range: {duration_s:.2f}s"
    assert torch.isfinite(audio).all(), "Audio contains non-finite values"


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_vibevoice_prefill_decode_mixed_batch_003(omni_runner: OmniRunner) -> None:
    """Regression: prefill+decode mixed batch must not crash."""
    long_prompt = (
        "This is a deliberately long prompt that will stay in the decode "
        "phase for many steps so that subsequent shorter prompts keep "
        "entering prefill alongside it, reproducing the prefill plus "
        "decode mixed batch scheduling pattern."
    )
    short_prompts = ["Hello one.", "Hello two.", "Hello three."]
    requests = [{"prompt": long_prompt}] + [{"prompt": p} for p in short_prompts]

    outputs = omni_runner.omni.generate(requests)
    assert len(outputs) == len(requests)

    for i, out in enumerate(outputs):
        audio = _extract_audio(out.outputs[0].multimodal_output)
        assert audio.shape[0] > 0, f"Request {i} produced empty audio"
        assert torch.isfinite(audio).all(), f"Request {i} has non-finite audio"

    # Assert no VibeVoice capability leaked into the shared AR runner.
    assert_non_vibevoice_ar_isolation(omni_runner)
