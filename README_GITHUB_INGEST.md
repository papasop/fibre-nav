# GPT-2 response-fibre evidence: GitHub ingest overlay

This add-only overlay is prepared for `papasop/neural-fibre-geometry`.

## Contents

- `evidence/gpt2_lora_r2_strict/`: frozen GPU R2 engine, Colab launcher, protocol, eight prospective seed records, node metrics, report and run log.
- `evidence/predictive_causal_boundaries/pc1/`: static predictive-capacity audit and complete results.
- `evidence/predictive_causal_boundaries/pc2/`: optimizer-access audit source. Its original results ZIP was not available when this overlay was built; see `RESULTS_REQUIRED.md`.
- `evidence/predictive_causal_boundaries/pc3/`: dynamic/history audit and complete results.
- `evidence/predictive_causal_boundaries/pc4/`: paired causal audit and complete results.
- `verify_evidence.py`: structural and decision-level verification.

## Scientific status

R2 is the strict functional confirmation: the task-related projected recorded update is compared with historical, time-misaligned, permuted, actual-update, and calibration-selected best-of-sixteen equal-norm random directions in the same instantaneous response kernel.

PC1-PC4 are failure-boundary audits. Their role is to prevent the R2 counterfactual opportunity result from being misreported as an optimizer-future law.

## Safe ingest

1. Extract this ZIP at the repository root.
2. Run `python evidence_release_overlay/verify_evidence.py` if the wrapper directory is retained, or `python verify_evidence.py` from inside it.
3. Re-run PC2 with its included Colab launcher and place the untouched result ZIP under `evidence/predictive_causal_boundaries/pc2/results/`.
4. Run `python verify_evidence.py --require-complete`; it must exit successfully before claiming PC1-PC4 are fully public.
5. Review, commit, push, and replace the paper's old commit SHA with the resulting public commit.

Do not rewrite the historical R1 decision. R2 is a separately frozen prospective protocol.

## R2 ZIP identity

The final uploaded R2 results ZIP has SHA-256:

`fc3239cc420c32d09868db05ec1dd6887d03433f6684b1b67d2ec1023ddfc3ba`

An earlier hash printed inside the run belongs to a pre-launcher archive state and must not be used as the public artifact hash.
