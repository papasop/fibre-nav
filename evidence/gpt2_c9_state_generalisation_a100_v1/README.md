# C9: GPT2_C9_STATE_GENERALISATION_A100_V1

**Recorded decision: C9_LOCAL_STATE_GENERALISATION_PASS.** Single-model-seed development evidence.
See [the Line C overview](../../docs/LINE_C_LEARNED_WRITER.md) for denominators and limitations.

- [Original recorded summary](recorded/SUMMARY.json)
- [Original source](source/)
- [Colab one-cell launcher](GPT2_C9_STATE_GENERALISATION_A100_V1.py)
- [Export provenance](EXPORT_PROVENANCE.json)
- [Original full-archive member manifest](recorded/MANIFEST.json)

This is a compact repository export. The recorded JSON files preserve original
bytes. The original MANIFEST.json describes the full result, including omitted
parent ZIPs and PT files; SHA256SUMS.json instead describes this export.
The root verifier checks both relationships.

Run from repository root:

    python3 scripts/verify_line_c_evidence.py --protocol-tests

This verifies recorded consistency and saved-head replay, without re-running GPT-2.
The A100 launcher requires an exact complete parent result ZIP or matching
complete resume ZIP. Neither this folder nor an NPZ weight export is a resume
input. Publish full results separately before claiming complete public reproduction.

Original full result SHA-256: f0f5abd2197311438bf23588dbcb1bedef1649380873b04f86b5e7b1340fce3b.
Original full result size: 284,981,674 bytes.

Raw 0/12 training and 0/8 new local states; external correction yields 12/12 and 8/8. The final NPZ is the actual frozen writer. Replay data contains saved features, preparations and raw states. Requested output bit is supplied; same four trained classes and access prompts. No controller dispatch or native update test.
