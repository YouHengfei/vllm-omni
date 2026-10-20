# VibeVoice multi-speaker reference-mapping demonstration

Evidence for the review comment on PR #9105 (hsliuustc0106, 2026-10-20):
the existing 2/4-speaker E2E repeats the same reference for every speaker
and checks only valid/non-silent audio, so it cannot demonstrate the
documented first-appearance reference-to-speaker mapping or the
unreferenced speaker's behavior. This directory provides that.

Generated at revision `4f14fec87c65f9634b7fc5682eab0d5204976ab4` on GPU7
(H100 80GB, deploy defaults: `max_num_seqs=4`, `max_model_len=16384`,
eager, FA3, seed 0, CFG 1.3, 10 diffusion steps, `max_new_tokens=512`).
Checkpoint `/SharedData/youhf/models/vibevoice_VibeVoice-1.5B-hf`.

## Setup

Two reference voices are committed test assets (reviewers can `git checkout`
the exact bytes):

- **refA** = `tests/assets/qwen3_tts/clone_2.wav`
- **refB** = `tests/assets/indextts2/ref_audio.wav`

Their raw WavLM-base-plus cosine SIM is 0.574. To let reviewers identify each
cloned voice, each turn was also generated as a single-speaker request with
its corresponding reference:

- `singleA_refA.wav` — turn 1 text cloned from refA (voice A, ~9.2 s)
- `singleB_refB.wav` — turn 2 text cloned from refB (voice B, ~10.1 s)

The two-speaker script (same for both cases, ~10 s per turn):

```text
Speaker 1: The first speaker welcomes the audience to the show and introduces the agenda for today's presentation, covering the recent progress on model adaptation and serving.
Speaker 2: The second speaker thanks everyone for coming and explains that the demonstration will cover voice cloning quality, multi-speaker handling, and the reference mapping behavior.
```

## Cases

### Case A — distinct references (`caseA_distinct_refs`)

`ref_audio=[refA, refB]`. Expected mapping per the renderer's first-appearance
rule: refA conditions speaker 1, refB conditions speaker 2.

- Request: `caseA_distinct_refs_request.json`
- Generated audio: `caseA_distinct_refs.wav` (16.5 s) — reviewers should hear
  voice A in the first turn and voice B in the second.

### Case B — partial reference (`caseB_partial_ref`)

`ref_audio=[refA]`. Expected: speaker 1 cloned from refA, speaker 2
unreferenced (native no-reference generation, voice sampled from diffusion
noise).

- Request: `caseB_partial_ref_request.json`
- Generated audio: `caseB_partial_ref.wav` (21.6 s) — reviewers should hear
  voice A in the first turn and a different, unreferenced voice in the second.

## Observed mapping

**Primary evidence: the audio files.** Reviewers should listen to
`singleA_refA.wav` and `singleB_refB.wav` to identify the two reference
voices, then to each multi-speaker output: case A's first turn matches voice
A and the second matches voice B; case B's first turn matches voice A and the
second turn is a different, unreferenced voice. The full multi-speaker audio
is the clean evidence; per-turn splits are not provided because the acoustic
decoder's overlap-add produces a blended transition at the speaker switch
(which a hard sample cut would split across both turns).

**Supplementary diagnostic (WavLM-base-plus cosine, a proxy — not speaker
verification).** A 0.67 s sliding-window SIM profile over each full output is
in `manifest.json`. WavLM clusters the two TTS-cloned voices at SIM 0.92 for
these longer samples, so per-window attribution is weak and must not be read
as proof; it is included only as a weak supporting diagnostic. Case A's
second half shows mildly elevated similarity to voice B (consistent with, but
not proof of, the refB→speaker 2 mapping); case B's second half shows no
clean voice-B switch, consistent with an unreferenced speaker 2.

## What this does and does not establish

- Establishes: the two cases run end to end; the request JSON and generated
  audio are reproducible from committed test assets; the expected mapping is
  documented and the full audio is available for inspection.
- Does not establish: statistical voice-distinctness or speaker-verification
  parity. VibeVoice-cloned voices from these references are similar in WavLM
  space (SIM 0.92); a stronger demonstration would need clearly gender/timbre
  distinct references and/or a speaker-verification model, which is out of
  scope for this evidence request.

See `manifest.json` for the full revision, checkpoint, references, per-window
SIM profile, and file list.
