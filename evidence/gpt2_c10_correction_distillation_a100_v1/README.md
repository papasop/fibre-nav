# C10: GPT2_C10_CORRECTION_DISTILLATION_A100_V1

**Recorded decision: C10_RAW_TRAINING_FUNCTION_FAILURE.** Single-model-seed development evidence.
See [the Line C overview](../../docs/LINE_C_LEARNED_WRITER.md) for denominators and limitations.

- [Original recorded summary](recorded/SUMMARY.json)
- [Original source](source/)
- [Colab one-cell launcher](GPT2_C10_CORRECTION_DISTILLATION_A100_V1.py)
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

Original full result SHA-256: 234228d37cf3d5e6ab1d2813b4610a5c14b03bf88b33c3a8ec8e8faaefa3ce53.
Original full result size: 318,008,911 bytes.

Raw acceptance 9/12 training and 2/8 new local states. Response drift causes failures. No online write Jacobian, SVD or correction; dataset preparation and offline training still use external geometry/autograd. The MSE-only head is exported for inspection, not an acceptance winner. The official failure decision is retained.
