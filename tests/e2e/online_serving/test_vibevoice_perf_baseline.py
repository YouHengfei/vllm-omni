# Phase A baseline: long-form VibeVoice generation via SSE, measure RTF/TTFA.
import base64
import json
import time
from pathlib import Path

import httpx
import pytest

from tests.helpers.media import get_asset_path, load_test_audio_data_url
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.advanced_model]

_MODEL_ROOT = "/SharedData/youhf/models"
_MODEL = str(Path(_MODEL_ROOT) / "VibeVoice")
_TOKENIZER = str(Path(_MODEL_ROOT) / "VibeVoice-1.5B-hf")
_REF = load_test_audio_data_url("qwen3_tts/clone_2.wav")

_SERVER = pytest.param(
    OmniServerParams(
        model=_MODEL,
        stage_config_path=get_deploy_config_path("vibevoice.yaml"),
        server_args=[
            "--tokenizer",
            _TOKENIZER,
            "--allowed-local-media-path",
            str(get_asset_path("").resolve()),
            "--disable-log-stats",
        ],
        env_dict={"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        init_timeout=900,
        stage_init_timeout=600,
        require_real_weights=True,
    ),
    id="official-vibevoice",
)

# A long English paragraph (~ many audio tokens).
LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "Sphinx of black quartz, judge my vow. "
    "How vexingly quick daft zebras jump. "
    "The five boxing wizards jump quickly. "
    "Bright vixens jump dozy fowl quack. "
    "Quick zephyrs blow, vexing daft Jim. "
    "Two driven jocks help fax my big quiz. "
    "Crazy Fredrick bought many very exquisite opal jewels. "
    "We promptly judged antique ivory buckles for the next prize. "
)


@pytest.mark.parametrize("omni_server", [_SERVER], indirect=True)
def test_vibevoice_perf_baseline_long_form_sse(omni_server) -> None:
    """Baseline: long-form SSE generation; record TTFA, total, RTF, token count."""
    url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
    payload = {
        "model": omni_server.model,
        "input": LONG_TEXT,
        "ref_audio": _REF,
        "response_format": "pcm",
        "stream": True,
        "timeout": 300.0,
    }
    t0 = time.perf_counter()
    ttfa = None
    total_pcm = bytearray()
    delta_count = 0
    done_event = None
    # trust_env=False: the SOCKS/HTTP proxy env vars must not intercept localhost.
    with httpx.Client(trust_env=False, timeout=300) as client, client.stream("POST", url, json=payload) as resp:
        assert resp.status_code == 200, resp.read().decode(errors="replace")
        assert resp.headers["content-type"].startswith("text/event-stream")
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line.removeprefix("data: "))
            if event["type"] == "speech.audio.delta":
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                total_pcm.extend(base64.b64decode(event["audio"]))
                delta_count += 1
            elif event["type"] == "speech.audio.done":
                done_event = event
    total = time.perf_counter() - t0
    assert done_event is not None
    n_samples = len(total_pcm) // 2  # s16le
    audio_dur = n_samples / 24000.0
    rtf = total / audio_dur if audio_dur > 0 else float("inf")
    n_tokens = n_samples // 3200
    print(
        f"\n[VIBEVOICE BASELINE] finish={done_event.get('finish_reason')} "
        f"ttfa={ttfa * 1000:.1f}ms total={total * 1000:.0f}ms "
        f"audio={audio_dur:.3f}s rtf={rtf:.3f} tokens={n_tokens} deltas={delta_count}"
    )
    assert n_tokens > 5
    assert rtf < 1.0
