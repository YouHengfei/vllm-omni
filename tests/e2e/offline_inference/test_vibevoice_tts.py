# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline E2E for VibeVoice TTS on the support-tecoomni (Plan-B) baseline.

Drives a real VibeVoice checkpoint through the single-stage AR pipeline and
verifies a well-formed 24 kHz mono waveform is produced. This exercises the
full downgraded flow: preprocess -> positive AR forward -> model-owned negative
KV branch -> CFG diffusion -> acoustic decode -> delta waveform output.

Environment variables:
    VIBEVOICE_TEST_MODEL     checkpoint dir (default /SharedData/youhf/models/VibeVoice)
    VIBEVOICE_TEST_TOKENIZER tokenizer repo (default Qwen/Qwen2.5-1.5B)
    VIBEVOICE_TEST_MAX_TOKENS decode cap (default 40)

Runs the engine in a spawned subprocess to isolate GPU/distributed state and to
satisfy the multiprocessing ``spawn`` guard. Requires a GPU and the checkpoint.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from queue import Empty
from typing import Any

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.tts]

_MODEL = os.getenv("VIBEVOICE_TEST_MODEL", "/SharedData/youhf/models/VibeVoice")
_TOKENIZER = os.getenv("VIBEVOICE_TEST_TOKENIZER", "Qwen/Qwen2.5-1.5B")
_MAX_TOKENS = int(os.getenv("VIBEVOICE_TEST_MAX_TOKENS", "40"))
_DEPLOY = "/tmp/vibevoice_e2e.yaml"
_REF = os.path.join(
    os.path.dirname(__file__),
    "../../../vllm_omni/model_executor/models/vibevoice/assets/default_0.wav",
)


def _e2e_worker(queue: Any) -> None:
    try:
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
        # flashinfer's sampling kernel needs a JIT toolchain (ninja) that is not
        # guaranteed on this baseline; the native sampler applies the same
        # allowed-token gate. This only affects the token sampler, not the model.
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

        import time

        import soundfile as sf
        import torch
        from transformers import AutoTokenizer
        from vllm import SamplingParams

        from vllm_omni.entrypoints.omni import Omni
        from vllm_omni.entrypoints.openai.tts_adapters.vibevoice import VibeVoiceTTSAdapter
        from vllm_omni.model_executor.models.vibevoice.pipeline import VIBEVOICE_VALID_TOKEN_IDS

        tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER)
        waveform, sample_rate = sf.read(_REF, dtype="float32")

        init_t0 = time.perf_counter()
        omni = Omni(
            model=_MODEL,
            stage_configs_path=_DEPLOY,
            tokenizer=_TOKENIZER,
            log_stats=False,
            init_timeout=1200,
        )
        init_s = time.perf_counter() - init_t0

        text = "Hello, this is a VibeVoice end to end test."
        rendered = VibeVoiceTTSAdapter._render_prompt([(0, text)], num_speakers=1)
        prompt = {
            "prompt": rendered,
            "prompt_token_ids": tokenizer.encode(rendered, add_special_tokens=False),
            "multi_modal_data": {"audio": [(waveform, sample_rate)]},
            "multi_modal_uuids": {"audio": ["vibevoice-e2e-0:audio:0"]},
        }
        sampling_params = [
            SamplingParams(
                max_tokens=_MAX_TOKENS,
                temperature=0.0,
                allowed_token_ids=list(VIBEVOICE_VALID_TOKEN_IDS),
                stop_token_ids=[151643],
                detokenize=False,
            )
        ]
        gen_t0 = time.perf_counter()
        outputs = omni.generate([prompt], sampling_params_list=sampling_params, use_tqdm=False)
        gen_s = time.perf_counter() - gen_t0

        mm = outputs[0].outputs[0].multimodal_output
        audio = mm.get("audio")
        if audio is None:
            audio = mm.get("model_outputs")
        chunks = audio if isinstance(audio, (list, tuple)) else [audio]
        chunks = [torch.as_tensor(c).detach().cpu().reshape(-1) for c in chunks if c is not None]
        wav = torch.cat(chunks) if chunks else torch.zeros(0)
        sr = mm.get("sr")
        sr = sr[-1] if isinstance(sr, (list, tuple)) else sr
        sr = int(sr.item()) if isinstance(sr, torch.Tensor) else int(sr)

        queue.put(
            {
                "ok": True,
                "init_s": init_s,
                "gen_s": gen_s,
                "sr": sr,
                "numel": int(wav.numel()),
                "dtype": str(wav.dtype),
                "finite": bool(torch.isfinite(wav).all()),
                "absmax": float(wav.abs().max()) if wav.numel() else 0.0,
                "rms": float(wav.pow(2).mean().sqrt()) if wav.numel() else 0.0,
            }
        )
    except Exception:
        queue.put({"ok": False, "traceback": traceback.format_exc()})


@pytest.mark.skipif(not os.path.isdir(_MODEL), reason=f"VibeVoice checkpoint not found at {_MODEL}")
def test_vibevoice_offline_tts_produces_waveform() -> None:
    import torch as _torch

    if not _torch.cuda.is_available():
        pytest.skip("requires a GPU")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_e2e_worker, args=(queue,))
    process.start()
    process.join(timeout=1500)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("VibeVoice E2E worker timed out")
    try:
        result = queue.get_nowait()
    except Empty:
        pytest.fail(f"worker exited without a result (exitcode={process.exitcode})")
    if not result.get("ok"):
        pytest.fail(result.get("traceback", "unknown worker failure"))

    # Well-formed 24 kHz mono waveform, a whole number of 3200-sample audio
    # tokens, finite, and non-trivial energy (real speech, not silence/clip).
    assert result["sr"] == 24000, result
    assert result["dtype"] == "torch.float32", result
    assert result["numel"] > 0 and result["numel"] % 3200 == 0, result
    assert result["finite"], result
    assert result["absmax"] > 0.01, result  # not silence
    duration_s = result["numel"] / result["sr"]
    assert 0.1 < duration_s < 60.0, result
