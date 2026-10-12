# C8: GPT2_C8_EXTRAPOLATION_DIAGNOSIS_A100_V1

**Recorded decision: C8_DEVELOPMENT_EXTRAPOLATION_DIAGNOSIS_COMPLETE.** Single-model-seed development evidence.
See [the Line C overview](../../docs/LINE_C_LEARNED_WRITER.md) for denominators and limitations.

- [Original recorded summary](recorded/SUMMARY.json)
- [Original source](source/)
- [Colab one-cell launcher](GPT2_C8_EXTRAPOLATION_DIAGNOSIS_A100_V1.py)
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

Original full result SHA-256: 92ce08de294a53a591920f8f52f15d2a1146c6d97b04e57e689221145fe40997.
Original full result size: 245,245,390 bytes.
