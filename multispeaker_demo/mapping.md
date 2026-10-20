# VibeVoice multi-speaker reference-mapping demonstration

Evidence for the review comment on PR #9105 (hsliuustc0106, 2026-10-20):
the existing 2/4-speaker E2E repeats the same reference for every speaker
and checks only valid/non-silent audio, so it cannot demonstrate the
documented first-appearance reference-to-speaker mapping or the
unreferenced speaker's behavior. This directory provides that.

Generated at revision `4f14fec87c65f9634b7fc5682eab0d5204976ab4` on GPU7
(H100 80GB, deploy defaults: `max_num_seqs=4`, `max_model_len=16384`,
eager, FA3, seed 0, CFG 1.3, 10 diffusion steps, `max_new_tokens=256`).
Checkpoint `/SharedData/youhf/models/vibevoice_VibeVoice-1.5B-hf`.

## Setup

Two reference voices are committed test assets (reviewers can `git checkout`
the exact bytes):

- **refA** = `tests/assets/qwen3_tts/clone_2.wav`
- **refB** = `tests/assets/indextts2/ref_audio.wav`

Their raw WavLM-base-plus cosine SIM is 0.574 (the most dissimilar pair
among available test assets). To obtain clean "voice A" and "voice B"
references for comparison, each turn was also generated as a single-speaker
request with its corresponding reference:

- `singleA_refA.wav` — turn 1 text cloned from refA (voice A)
- `singleB_refB.wav` — turn 2 text cloned from refB (voice B)

The two-speaker script (same for both cases):

```text
Speaker 1: The first speaker welcomes the audience to the show.
Speaker 2: The second speaker thanks everyone for coming today.
```

## Cases

### Case A — distinct references (`caseA_distinct_refs`)

`ref_audio=[refA, refB]`. Expected mapping per the renderer's first-appearance
rule: refA conditions speaker 1, refB conditions speaker 2.

- Request: `caseA_distinct_refs_request.json`
- Full audio: `caseA_distinct_refs.wav` (5.87 s)
- Turn split at the midpoint (2.93 s): `caseA_distinct_refs_turn1.wav`,
  `caseA_distinct_refs_turn2.wav`

### Case B — partial reference (`caseB_partial_ref`)

`ref_audio=[refA]`. Expected: speaker 1 cloned from refA, speaker 2
unreferenced (native no-reference generation, voice sampled from diffusion
noise).

- Request: `caseB_partial_ref_request.json`
- Full audio: `caseB_partial_ref.wav` (7.60 s)
- Turn split at the single-speaker refA duration (3.33 s):
  `caseB_partial_ref_turn1.wav`, `caseB_partial_ref_turn2.wav`

## Observed mapping

**Primary evidence: the audio files.** Reviewers should listen to
`singleA_refA.wav` and `singleB_refB.wav` (the two reference voices), then to
each multi-speaker output: case A's first turn matches voice A and the second
matches voice B; case B's first turn matches voice A and the second turn is a
different, unreferenced voice.

**Supplementary diagnostic (WavLM-base-plus cosine, a proxy — not speaker
verification).** The SIM profile was computed over 0.67 s sliding windows.
WavLM clusters the two TTS-cloned voices at SIM 0.78, so per-turn attribution
is weak and must not be read as proof:

| Case | Second-half mean SIM to voice A | Second-half mean SIM to voice B | Voice B dominant? |
| --- | --- | --- | --- |
| A (distinct refs) | 0.576 | 0.723 | yes (+0.15) |
| B (partial ref) | 0.559 | 0.621 | weakly (+0.06) |

Case A's second half (turn 2) shows elevated similarity to voice B, consistent
with the refB→speaker 2 mapping. Case B's second half does not show a
comparably clean voice-B switch, consistent with an unreferenced speaker 2;
the small margin is within the proxy's noise for these similar cloned voices.
The full per-window profile is in `manifest.json`.

## What this does and does not establish

- Establishes: the two cases run end to end; the request JSON and generated
  audio are reproducible from committed test assets; the expected mapping is
  documented and the audio is available for inspection.
- Does not establish: statistical voice-distinctness or speaker-verification
  parity. VibeVoice-cloned voices from these references are similar in WavLM
  space (SIM 0.78); a stronger demonstration would need clearly gender/timbre
  distinct references and/or a speaker-verification model, which is out of
  scope for this evidence request.

See `manifest.json` for the full revision, checkpoint, references, per-window
SIM profile, and file list.
