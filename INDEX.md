# VibeVoice PR1 evidence index

Evidence published separately from model PR #8909. Raw historical artifacts remain available;
new batches are appended rather than overwriting earlier reports.

| Batch | Entry point | Status |
| --- | --- | --- |
| 2026-10-15 generation plus CPU quality supplement | [Audited evidence](batches/ab_std_20261015_080817/EVIDENCE.md) | Eight generation configurations; ZH Omni B4 quality replaced by a full CPU re-evaluation; provenance/isolation/launcher limitations documented |
| 2026-10-14 historical A/B | [Historical evidence](EVIDENCE.md) | Retrospective manifest; retained unchanged |

The latest batch's [audited manifest](batches/ab_std_20261015_080817/results/manifest.json) and
[effective results](batches/ab_std_20261015_080817/provenance/effective_results.json) identify the
quality and generation sources separately. Its raw child manifests are empty because of a launcher
metadata bug; they are preserved, not presented as proof of automated completion.

Checksums are scoped per batch: root `SHA256SUMS` covers the original historical bundle;
`batches/ab_std_20261015_080817/SHA256SUMS` covers the new batch. Root `INDEX.md` is the navigation
entry point; use fixed Git commit URLs when citing these materials.

No published evidence update changes the model PR's source, recipe, PR body or review status.
