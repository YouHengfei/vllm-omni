# VibeVoice Bundled Voice Asset Provenance Audit

> **Audit date:** 2026-08-22
>
> **Status:** `UNRESOLVED — NOT READY FOR UPSTREAM REDISTRIBUTION`
>
> **Scope:** `vllm_omni/model_executor/models/vibevoice/assets/default_{0..3}.wav`

## Conclusion

No primary-source record currently establishes who created or recorded the four
WAV files, where they came from, which license applies to them, whether they
contain a real person's voice, or whether redistribution and voice-cloning
consent were granted.

The files can remain on the preserved development/remediation branch while the
question is investigated, but they must not be represented as
“framework-owned” or included in a final upstream submission without evidence.
If the evidence listed below cannot be produced, the final submission should
remove the files and require explicit `ref_audio` or an uploaded voice.

This conclusion is intentionally narrower than a copyright determination. An
absence of evidence is not proof that redistribution is forbidden; it is also
not evidence that redistribution is allowed.

## Inventory

| File | SHA-256 | Git blob | Duration | Encoding |
| --- | --- | --- | ---: | --- |
| `default_0.wav` | `7dfbfe7061982d0f91997bbc0b8593e816e50aab8a7c1158620e6043ddd2c1b7` | `37b14d09664e425dc75e554cc3c56f7fc82e8093` | 9.042250 s | PCM s16le, mono, 24 kHz |
| `default_1.wav` | `4aeab909c89e0617f8a339973f463a4467c61492572694607df27b89c04dd2c5` | `82e26d41de7cc2378dd57bd125599017ff182691` | 7.700000 s | PCM s16le, mono, 24 kHz |
| `default_2.wav` | `f3161536a6dac1b9fbb3aac7894c5cc43fff75e58e1df9e57064069267e16be3` | `4a67e844abe8bb60ca4fa04dcad1d617167ba6d7` | 5.469167 s | PCM s16le, mono, 24 kHz |
| `default_3.wav` | `8de5520c8e54b3b1d420f15e23be89b2d16d9c0a2bd6edb06075816e53b33876` | `66fb4a82b02760e8ec64e2f1de93791a61b3a63f` | 8.080000 s | PCM s16le, mono, 24 kHz |

Each file contains only the minimal RIFF `fmt` and `data` chunks. `ffprobe`
reported no title, artist, comment, copyright, encoder, source, or other format
tag that could identify its origin.

The audit deliberately did not infer speaker identity, demographic attributes,
or consent from the sound of the recordings.

## Repository history

All four files first appear together in:

- commit [`e0290cbc33054596380565ad56fd65aa934dd69e`](https://github.com/YouHengfei/vllm-omni/commit/e0290cbc33054596380565ad56fd65aa934dd69e);
- author/committer: `YouHengfei <474029121@qq.com>`;
- message: `feat:add defauil ref audio for only text input`.

The commit adds the binary files and describes fallback behavior, but does not
record a source URL, source revision, creator, speaker, transcript, license,
consent, generation method, or attribution. The commit has no Git note carrying
that information. Earlier history on the preserved branch contains no version
of these paths.

A commit author records who added a file to this repository; it does not by
itself establish authorship of the recording or the voice represented in it.

## Primary-source searches

### Microsoft VibeVoice source repository

The recursive first-party GitHub tree at revision
[`94da20d98b2fa7688e9cbfaf7692ddb4954f7600`](https://api.github.com/repos/microsoft/VibeVoice/git/trees/94da20d98b2fa7688e9cbfaf7692ddb4954f7600?recursive=1)
contains no `default_0.wav` through `default_3.wav` and no raw TTS reference WAV
with a matching path or checksum. It does contain separately named ASR demo
media and serialized Realtime voice presets. No claim is made that those files
are related to the four audited WAVs.

The repository's MIT
[`LICENSE`](https://github.com/microsoft/VibeVoice/blob/303b2833e01cff4578ec278bbfe536da54bd19fe/LICENSE)
applies to material distributed by that repository under its terms. Because the
four audited files were not found there and have no recorded derivation from
it, that license cannot be assigned to them by inference.

### Microsoft VibeVoice-1.5B model repository

The first-party Hugging Face tree at model revision
[`c00898d257e6b46004e3e2866a47534085fb685a`](https://huggingface.co/api/models/microsoft/VibeVoice-1.5B/tree/c00898d257e6b46004e3e2866a47534085fb685a?recursive=true&expand=false)
contains model/configuration files and no WAV/MP3/FLAC voice asset. Its
[`README.md`](https://huggingface.co/microsoft/VibeVoice-1.5B/blob/c00898d257e6b46004e3e2866a47534085fb685a/README.md)
does not identify or license these four recordings.

The model card is directly relevant to the consent gate. It says that voice
impersonation without explicit, recorded consent is outside intended use and
that users are responsible for sourcing data legally and ethically, including
securing appropriate rights. A model-repository MIT label does not establish
those facts for an unidentified recording added elsewhere.

### Local primary-source checkout

The local `microsoft/VibeVoice` checkout at
`303b2833e01cff4578ec278bbfe536da54bd19fe` and its reachable history were
searched for tracked WAV/MP3/FLAC paths and for the audited checksums. No match
was found. This local result corroborates the immutable first-party tree queries
above but is not treated as a substitute for them.

## Evidence required to resolve the gate

For each file, retain a durable record of:

1. immutable source URL and source revision;
2. original filename and checksum;
3. recording or generation method;
4. recording author/producer and copyright holder;
5. applicable asset license and required attribution;
6. permission to redistribute the recording in this repository;
7. whether the voice is synthetic or belongs to a real person;
8. if real, explicit recorded consent for this redistribution and voice-cloning
   use;
9. any geographic, research-only, or commercial-use restrictions;
10. a maintainer/legal decision that the evidence is sufficient.

An email or issue response should be archived or linked from this document; a
verbal statement or filename such as `default_0` is insufficient.

The first-party model card gives `VibeVoice@microsoft.com` as the contact for
questions and undesired behavior. Contacting that address may clarify whether
Microsoft published a suitable, licensed reference set, but it cannot identify
the current files without their original source information.

## Final-submission decision rule

- **Evidence complete:** add an asset manifest with the facts above, required
  attribution, checksums, and the approving decision; then retain only the
  files covered by that evidence.
- **Evidence incomplete:** remove all four WAVs, remove the bundled-default
  fallback, require explicit reference audio/uploaded voice, and update tests
  and user documentation.

No quality or convenience result overrides this gate.
