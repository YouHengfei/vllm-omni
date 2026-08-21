# VibeVoice

vLLM-Omni supports the official `microsoft/VibeVoice-1.5B` text-to-speech
checkpoint without converting its weights. The integration targets inference
only; VibeVoice Realtime, ASR, training, and other model sizes are not included.

## Capabilities

- OpenAI-compatible `POST /v1/audio/speech`
- 24 kHz mono waveform output (`wav` or signed 16-bit `pcm`)
- Streaming PCM and structured SSE streaming
- One to four speakers, using four bundled default reference voices when
  `ref_audio` and `voice` are omitted
- Explicit reference audio per speaker and uploaded voices through
  `/v1/audio/voices`
- Classifier-free guidance with the checkpoint's positive and negative Qwen2
  branches
- Acoustic and semantic feedback after every generated audio token
- Single-GPU deployment by default; TP=2 is supported as an optional capability
  topology

Each generated audio token contains 3,200 samples, or approximately 133.3 ms at
24 kHz. `audio_eos_token_id` ends one audio segment; only the request EOS token
ends the request.

## Start the server

The default deploy configuration is `vllm_omni/deploy/vibevoice.yaml`. It uses
one GPU, tensor parallel size 1, `max_num_seqs=4`, and fixed 8 GiB positive and
negative KV-cache pools.

```bash
vllm serve microsoft/VibeVoice-1.5B \
  --omni \
  --host 0.0.0.0 \
  --port 8000 \
  --tokenizer Qwen/Qwen2.5-1.5B
```

The tokenizer can be omitted when the checkpoint's
`preprocessor_config.json.language_model_pretrained_name` is available. In an
offline deployment, pre-cache that tokenizer or pass its local path explicitly.

To use reference audio from a server-local file, grant access only to the
containing directory:

```bash
vllm serve microsoft/VibeVoice-1.5B \
  --omni \
  --host 0.0.0.0 \
  --port 8000 \
  --tokenizer Qwen/Qwen2.5-1.5B \
  --allowed-local-media-path /srv/vibevoice-references
```

## Default reference voices

VibeVoice includes four framework-owned reference voices. When both `ref_audio`
and `voice` are omitted, the adapter assigns the first N defaults to the N
speakers in first-appearance order. Plain text therefore uses default voice 0,
and a four-speaker script uses all four defaults.

Explicit references remain all-or-nothing: once `ref_audio` is provided, its
length must exactly match the number of speakers. A partial list is rejected
instead of silently mixing user and default voices.

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --output default-voice.wav \
  -d '{
    "model": "microsoft/VibeVoice-1.5B",
    "input": "Hello from the bundled default voice.",
    "response_format": "wav",
    "max_new_tokens": 1024
  }'
```

## Single-speaker synthesis

To select a custom voice, provide one reference audio as a data URL, an allowed
`file://` URI, or an uploaded voice. Omitting both selects bundled default voice
0.

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --output speech.wav \
  -d '{
    "model": "microsoft/VibeVoice-1.5B",
    "input": "Hello from VibeVoice.",
    "ref_audio": "file:///srv/vibevoice-references/speaker.wav",
    "response_format": "wav",
    "max_new_tokens": 1024
  }'
```

For a non-streaming response, `X-Finish-Reason` is `stop` when generation ended
naturally and `length` when `max_new_tokens` was reached.

## SSE streaming

Set both `stream=true` and `stream_format="sse"` to receive
`speech.audio.delta` events followed by one `speech.audio.done` event. Delta
audio is base64-encoded signed 16-bit mono PCM at 24 kHz.

```bash
curl -N http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "microsoft/VibeVoice-1.5B",
    "input": "This response is streamed.",
    "ref_audio": "file:///srv/vibevoice-references/speaker.wav",
    "response_format": "pcm",
    "stream": true,
    "stream_format": "sse",
    "max_new_tokens": 1024
  }'
```

The terminal event includes `finish_reason`. Applications that evaluate output
quality should aggregate only samples with `finish_reason="stop"`.

Raw PCM streaming remains available with `stream_format="audio"`.

## Multiple speakers

Use `Speaker N:` lines and pass reference audios in first-appearance order.
VibeVoice supports at most four distinct speakers and exactly one reference per
speaker.

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --output conversation.wav \
  -d '{
    "model": "microsoft/VibeVoice-1.5B",
    "input": "Speaker 0: Welcome.\nSpeaker 1: Thank you.\nSpeaker 0: Let us begin.",
    "ref_audio": [
      "file:///srv/vibevoice-references/speaker-0.wav",
      "file:///srv/vibevoice-references/speaker-1.wav"
    ],
    "response_format": "wav",
    "max_new_tokens": 2048
  }'
```

## Uploaded voices

A voice uploaded through `POST /v1/audio/voices` can be selected by its `voice`
name instead of passing `ref_audio` on every request. `voice` and `ref_audio`
are mutually exclusive. Uploaded `voice` names are single-speaker only.
Multi-speaker requests can either omit both fields to use the bundled defaults,
or pass a complete reference list directly.

## Generation defaults

The bundled deployment matches the official generation behavior:

```text
temperature=0
top_p=1.0
top_k=-1
repetition_penalty=1.0
guidance_scale=1.3
num_diffusion_steps=10
```

The model constrains sampling to VibeVoice's audio BOS, audio token, audio EOS,
and request EOS tokens. The default checkpoint limit is 40,500 generated
tokens; applications should pass a smaller `max_new_tokens` when a bounded
request is required.

## Optional TP=2 deployment

TP=1 is recommended for this 1.5B model because TP=2 communication can increase
latency. TP=2 remains supported for weight sharding and capability validation.
Create a local overlay such as `/etc/vllm-omni/vibevoice-tp2.yaml`:

```yaml
base_config: /path/to/vllm_omni/deploy/vibevoice.yaml

stages:
  - stage_id: 0
    devices: "0,1"
    tensor_parallel_size: 2
    kv_cache_memory_bytes: 6442450944  # 6 GiB per rank
    engine_extras:
      additional_config:
        vibevoice_runtime_config:
          negative_kv_cache_memory_bytes: 4294967296  # 4 GiB per rank
```

Then launch with:

```bash
vllm serve microsoft/VibeVoice-1.5B \
  --omni \
  --tokenizer Qwen/Qwen2.5-1.5B \
  --deploy-config /etc/vllm-omni/vibevoice-tp2.yaml
```

TP=2 is a supported topology, not the default performance baseline.

## Tests

CPU and contract tests do not require model weights:

```bash
pytest -q tests/model_executor/models/vibevoice \
  tests/worker/test_gpu_ar_model_runner.py
```

Official-weight offline and OpenAI Speech tests require an H100 and an explicit
advanced run level:

```bash
VIBEVOICE_TEST_MODEL=microsoft/VibeVoice-1.5B \
VIBEVOICE_TEST_TOKENIZER=Qwen/Qwen2.5-1.5B \
pytest -q -s --run-level advanced_model \
  tests/e2e/offline_inference/test_vibevoice_tts.py \
  tests/e2e/online_serving/test_vibevoice_tts.py
```

The portable DFX configuration is
`tests/dfx/stability/tests/test_vibevoice.json`. Performance, exhaustive TP=2,
and multi-scenario long-duration tests are intentionally maintained outside the
regular merge gate.

## Limitations

- Only `microsoft/VibeVoice-1.5B` TTS is supported.
- Realtime/duplex inference, ASR, and training are not supported.
- Audio output is fixed to 24 kHz mono.
- A maximum of four speakers is accepted.
- Reference text and model-specific fields from other TTS APIs are rejected.
- TP=1 is the recommended deployment for latency and per-GPU throughput.
