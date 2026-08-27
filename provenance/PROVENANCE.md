# Provenance

Author: Y.Y.N. Li

Paper context: *The Geometry of Functional Freedom in Neural Networks*

## CNER-F v16

The first Colab result archive is preserved verbatim as
`evidence/confirmed/f16_v16/raw/cner_cnn_mnist_fisher_confirm_v16_results_raw.zip`.

Its SHA-256 is:

`57de70d2a8d722679713539ff3110ddb72fe5aac23a45ceddd49381cc6cb089e`

The original archive contained two stale, non-input artifacts named
`seed_summary.csv` and `action_comparison.png`. The CSV contains seeds from an
earlier experiment. Neither file was read by the v16 computation or used by its
decision rule. The authoritative v16 outputs in that archive are `result.json`
and `REPORT.md`.

The files under `evidence/confirmed/f16_v16/results/` are a deterministic
archival rendering of the unchanged v16 `result.json`: the report and result are
byte-preserved, the frozen protocol is included explicitly, and the v16-specific
CSV and plot are regenerated solely from the result record. No seed, action,
endpoint, gate, p-value, or scientific status is recomputed or altered.

## Moving fibre evidence v3.1

Bundle name: `cner_moving_fibre_github_evidence_v3_1`

Created UTC: 2026-08-27

The files under stage `raw/` directories are copied without modification from
the original returned result ZIPs. Extracted results are provided for review
convenience.

| Stage | Protocol | Status |
|---|---|---|
| v3.0b | `CNER_CNN_MNIST_MOVING_F16_CONFIRM_V3_0B` | `MOVING_F16_V30B_CONFIRMED_RESTRICTED_ORDERING` |
| v3.1a | `CNER_CNN_MNIST_MOVING_FIBRE_QUICK_V3_1A` | `MOVING_FIBRE_V31A_QUICK_CANDIDATE_SUPPORTED` |
| v3.1b | `CNER_CNN_MNIST_MOVING_FIBRE_DEPTH_V3_1B` | `MOVING_FIBRE_V31B_DEPTH_CANDIDATE_NOT_SUPPORTED` |
| v3.1c | `CNER_CNN_MNIST_REALISABILITY_COST_V3_1C` | `REALISABILITY_COST_V31C_SCALING_CANDIDATE_SUPPORTED` |
