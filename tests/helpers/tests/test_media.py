# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from tests.helpers import media

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _stub_transcribe(monkeypatch) -> dict:
    """Capture the kwargs the helper hands to whisper, with no model and no device."""
    captured: dict = {}

    class FakeModel:
        def transcribe(self, path, **kwargs):
            captured.update(kwargs)
            return {"text": "London"}

    monkeypatch.setitem(sys.modules, "whisper", SimpleNamespace(load_model=lambda size, device=None: FakeModel()))
    # Keep the unit test off the accelerator probe so it stays hermetic.
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.platforms",
        SimpleNamespace(current_omni_platform=SimpleNamespace(is_available=lambda: False)),
    )
    return captured


def test_english_text_preprocessing_does_not_require_opencc(monkeypatch):
    monkeypatch.setitem(sys.modules, "opencc", None)

    assert media.preprocess_text("The weather is nice today!") == ("the weather is nice today")


def test_ffmpeg_fallback_exposes_imageio_binary(monkeypatch, tmp_path):
    target = tmp_path / "ffmpeg-imageio"
    target.write_bytes(b"test-only")
    monkeypatch.setattr(media.shutil, "which", lambda _name: None)
    monkeypatch.setattr(media.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        SimpleNamespace(get_ffmpeg_exe=lambda: str(target)),
    )

    media._ensure_test_ffmpeg_on_path()

    link = tmp_path / "vllm_omni_test_bin" / "ffmpeg"
    assert link.resolve() == target.resolve()
    assert str(link.parent) == media.os.environ["PATH"].split(media.os.pathsep)[0]


def test_transcribe_forwards_requested_language(monkeypatch):
    captured = _stub_transcribe(monkeypatch)

    media._whisper_transcribe_in_current_process("/tmp/does-not-matter.wav", "small", language="en")

    assert captured["language"] == "en"


def test_transcribe_defaults_to_auto_language(monkeypatch):
    # Auto-detect must stay the default: forcing a language globally would break
    # the non-English audio tests (e.g. the Chinese Qwen3-Omni prompts).
    captured = _stub_transcribe(monkeypatch)

    media._whisper_transcribe_in_current_process("/tmp/does-not-matter.wav", "small")

    assert captured.get("language") is None


def test_bytes_entrypoint_forwards_language_to_subprocess(monkeypatch, tmp_path):
    """Cover the two hops the tests above skip: bytes -> file -> executor.submit.

    The real call crosses a spawn ProcessPoolExecutor, so capture what gets
    submitted rather than what the worker eventually does.
    """
    submitted: dict = {}

    class FakeExecutor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, fn, *args):
            submitted["fn"] = fn
            submitted["args"] = args
            return SimpleNamespace(result=lambda: "London")

    monkeypatch.setattr(media.concurrent.futures, "ProcessPoolExecutor", lambda *a, **kw: FakeExecutor())

    wav = tmp_path / "clip.wav"
    sf.write(wav, np.zeros(2400, dtype=np.float32), 24000)

    assert media.convert_audio_bytes_to_text(wav.read_bytes(), "small", "en") == "London"

    assert submitted["fn"] is media._whisper_transcribe_in_current_process
    assert submitted["args"][1:] == ("small", "en")
