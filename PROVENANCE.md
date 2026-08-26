# CNER-F v16 provenance and archive correction

The first Colab result archive is preserved verbatim as
`evidence/raw/cner_cnn_mnist_fisher_confirm_v16_results_raw.zip`.

Its SHA-256 is:

`57de70d2a8d722679713539ff3110ddb72fe5aac23a45ceddd49381cc6cb089e`

The original archive contained two stale, non-input artifacts named
`seed_summary.csv` and `action_comparison.png`. The CSV contains seeds from an
earlier experiment. Neither file was read by the v16 computation or used by
its decision rule. The authoritative v16 outputs in that archive are
`result.json` and `REPORT.md`.

The embedded protocol SHA-256 in the authoritative result is:

`dbe91a8572f40e5bdfe650cf6602a8878bd160153019fa12a5d35b2ea65771eb`

The files under `evidence/corrected/` are a deterministic archival rendering
of the unchanged v16 `result.json`: the report and result are byte-preserved,
the frozen protocol is included explicitly, and the v16-specific CSV and plot
are regenerated solely from the result record. No seed, action, endpoint,
gate, p-value, or scientific status is recomputed or altered.

The executable launcher is corrected prospectively to clear its output
directory and to package only a fixed whitelist of v16 outputs. This packaging
repair does not change the frozen scientific protocol.
