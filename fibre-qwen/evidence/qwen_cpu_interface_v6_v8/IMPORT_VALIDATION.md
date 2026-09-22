# Import validation

- 20 existing V8 engineering tests passed in the repository directory layout.
- All imported Python files passed syntax parsing.
- V8 frozen manifest: all 19 entries matched the delivered source; ZIP contents matched extracted source byte for byte.
- Original V8 ZIP SHA-256: `2cadfe7200470e85200aed1195e400cbbe600a193167fd761238f70143797470`.
- V6 original results ZIP: CRC and all 86 result-manifest hashes verified; 64 records present; original failed verdict preserved.
- V7: 96 log points and the reported failed summary parsed; first-pass bounded counts independently recomputed from visible outputs.
- V8: 18 partial log points parsed; parsed fields and visible Boolean outputs checked; no final results inferred.
- Dedicated imported-payload manifest and updated provenance entries verified against local bytes.

These are engineering and archival checks. The pretrained Qwen experiment was not rerun for this import. V7/V8 raw result ZIPs were not available; their score and numerical gates cannot be independently audited from these logs. The earlier small random-Qwen engineering check is documented in the unchanged source VALIDATION.md and is not a pretrained-model result.
