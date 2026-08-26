# VibeVoice

> Zero-shot multi-speaker text-to-speech with reference audio cloning

## Summary

- Vendor: Microsoft
- Model: [`microsoft/VibeVoice-1.5B`](https://huggingface.co/microsoft/VibeVoice-1.5B)
- Task: text2speech (voice cloning via reference audio)
- Mode: Online serving (OpenAI-compatible `/v1/audio/speech`)
- Maintainer: Community

## When to use this recipe

Use this recipe to serve VibeVoice-1.5B on a single H100 GPU via vLLM-Omni.
VibeVoice is a 1.5B AR + diffusion TTS model that clones any speaker's voice
from a short reference audio clip (≤60 s). It supports up to four speakers per
request with independent references and outputs 24 kHz mono PCM.

## Prerequisites

- 1× H100 80 GB (or equivalent GPU with ≥30 GB free)
- `transformers >= 5.10.1, < 5.15` (VibeVoice model classes)
- Local checkpoint: `microsoft/VibeVoice-1.5B`
- Tokenizer: `Qwen/Qwen2.5-1.5B` (or the HF-converted checkpoint's tokenizer)

## Serve

```bash
vllm serve microsoft/VibeVoice-1.5B \
  --omni \
  --tokenizer Qwen/Qwen2.5-1.5B \
  --host 127.0.0.1 \
  --port 8000
```

The default deploy config (`vllm_omni/deploy/vibevoice.yaml`) sets:

- TP=1, `max_num_seqs=4`, `max_model_len=65536`
- Positive/negative KV cache: 8 GiB each
- Diffusion + decode CUDA graphs enabled
- Greedy AR sampling (`temperature=0.0`)

## Usage

### Single speaker with reference audio

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/VibeVoice-1.5B",
    "input": "Hello, this is a test of zero-shot voice cloning.",
    "ref_audio": "data:audio/wav;base64,<base64-encoded-wav>",
    "response_format": "wav"
  }' --output speech.wav
```

### Streaming SSE with finish_reason

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/VibeVoice-1.5B",
    "input": "Streaming text to speech.",
    "ref_audio": "data:audio/wav;base64,<base64-encoded-wav>",
    "response_format": "pcm",
    "stream": true,
    "stream_format": "sse"
  }'
```

### Four speakers with bundled defaults

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/VibeVoice-1.5B",
    "input": "Speaker 0: Hello.\nSpeaker 1: Welcome.\nSpeaker 2: Let us begin.\nSpeaker 3: Thank you.",
    "response_format": "wav"
  }' --output speech.wav
```

When `ref_audio` is omitted entirely, four bundled Apache-2.0 reference voices
are assigned in speaker first-appearance order. See
[ASSET_PROVENANCE.md](../../docs/design/vibevoice/ASSET_PROVENANCE.md) for
the source and license of each bundled reference.

## Key behaviors

- **No request-level seed**: VibeVoice uses greedy AR sampling and a global
  diffusion RNG. The `seed` field is rejected; omit it.
- **Audio EOS ≠ request EOS**: `audio_eos_token_id=151653` ends a speaker
  segment; `eos_token_id=151643` ends the request. `finish_reason="stop"`
  means natural completion; `finish_reason="length"` means valid truncation.
- **Reference audio limit**: 60 s max per reference, one per speaker.
- **Output**: 24 kHz mono, 3200 samples per token.

## References

- Model card: <https://huggingface.co/microsoft/VibeVoice-1.5B>
- vLLM-Omni docs: [docs/models/vibevoice.md](../../docs/models/vibevoice.md)
- Deploy config: `vllm_omni/deploy/vibevoice.yaml`
