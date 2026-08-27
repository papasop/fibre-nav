# Provenance and methodological history

This archive is assembled from frozen executable packages and their original
result ZIPs. Extracted result directories are convenience renderings; the raw
ZIPs remain the authoritative byte-preserved outputs.

The F16 v16 subtree preserves its original contamination disclosure: two stale
non-input presentation artifacts in the first Colab archive were replaced by
deterministic renderings of the unchanged authoritative `result.json`. See
`evidence/confirmed/f16_v16/PROVENANCE.md`.

v3.1d used 16 seeds 69726--69741 and failed its formal total gate because only
6/16 fixed arms met a high log--log R-squared threshold. The fixed costs were
nearly constant, as predicted, making R-squared ill-conditioned through a
near-zero total variance denominator. The v3.1d code, report, and raw results
are preserved under `developmental/realisability_cost_v3_1d/`.

Before observing v3.1e outcomes, the fixed-arm veto was replaced by frozen
tests `abs(exponent) <= 0.25` and four-point relative span `<= 0.20`; the moving
arm retained exponent `>= 0.50` and log--log R-squared `>= 0.80`. v3.1e used
new seeds 70726--70741. Seed 70741 was excluded by the predeclared source
accuracy gate (0.898 < 0.900); all 15 comparable seeds passed every substantive
gate.

No result has been relabelled by deleting a failed predecessor. No GPT-2 code
is included in this release because GPT-2 currently supplies a boundary result,
not a confirmed capacity-weighted CNER or moving-fibre replication.

## v1.2 addition: Moving-Fibre F16

v3.2a first placed six causal algorithms in the same moving-current-kernel
domain. It supported a restricted action candidate but showed that natural
gradient was not the least-retraction-cost path. v3.2b froze four radii and
passed its four-seed developmental discretization gate in 3/4 seeds.

Before observing v3.2c, the confirmation protocol froze 16 new seeds
73726--73741, at least 14 fully comparable seeds, at least 12 seeds satisfying
all positive gates, and at most two natural wins under the wrong metric. The
result retained 14 fully comparable seeds. All 14 satisfied every positive
gate and natural won under the wrong metric in 0/14.

Seeds 73730 and 73732 were excluded because the coarse-radius natural arm's
wrong-metric left-versus-trapezoid change exceeded 0.08. No post-result
relaxation was applied. The result also retained the predeclared descriptive
cost branch: natural gradient's smallest-radius cost remained at least 1.5
times the minimum in 14/14 comparable seeds.
