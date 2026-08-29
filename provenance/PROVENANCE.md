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

No result has been relabelled by deleting a failed predecessor. GPTW v1.1.0
now preserves the complete restricted GPT-2/LoRA development and confirmation
chain, including the formally unsuccessful initial audit and its dtype repair.
It does not supply a capacity-weighted CNER replication or a full-model result.

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

## v1.2.2 formula and robustness audit

This release does not alter the frozen v3.1e or v3.2c executables, protocols,
raw archives, eligibility decisions, or primary outcomes. It adds the explicit
discrete definition of \(S_{\rm MF16}\), an audit note separating Euclidean
implementation identities from their general-metric forms, and a
post-confirmation leave-one-radius-out analysis reconstructed directly from
the archived v3.1e seed JSON files.

The LOO analysis was not part of the original prospective gate and is labelled
accordingly. It strengthens finite-radius robustness without changing the
claim boundary. The original v3.2c engine SHA-256 remains
`8b3ba6f37bc57bad5b3752238831366c668f94bf7959c2f45cf5749d016cbb52`.

## v1.3.0 external functional confirmation

ResNet v4.0a used seeds 61726--61729 and remains formally unsupported: its
candidate gate improperly treated the unconstrained ambient gradient as a
feasible competition arm. v4.0a-r1 repaired the comparison class on new seeds
62726--62729 and is retained as a development candidate.

v4.0b used seeds 63726--63741 in the complete final-classifier parameter space.
All functional and effect-size gates passed at every anchor count, but the
float32 finite-response leakage gate failed, so the run remains formally
unconfirmed. v4.0c prospectively froze new seeds 64726--64741, float64 response
geometry, a dimensionless kernel-residual gate and a finite-response gate while
leaving all functional gates unchanged. Its first implementation, v4.0c-r0,
terminated before any seed result because matched random vectors defaulted to
float32 while the response row basis was float64. Revision r1 changed only the
random-vector dtype and added diagnostic logging; it did not change scientific
seeds, data, thresholds, controls or gates.

v4.0c-r1 passed every gate in 16/16 seeds at all three anchor counts. This
confirms selected tangent value in the complete final classifier of a frozen
ResNet-18 representation. It does not confirm ResNet-backbone navigation,
moving-fibre transport, realizability-cost scaling or Moving-Fibre F16.

Authoritative hashes:

- v4.0c-r1 engine: `dca49cb187c59a2ac8e3e55fe4970349f258fed36b7ffbfca510770c19af9f63`
- v4.0c-r1 raw results ZIP: `a3edf61ed8fcd029f302f922650071dec9a1b60c501ace34a4ec6965ebf71823`

## v1.4.0 GPTW restricted cross-modal confirmation

The GPTW snapshot was imported byte-for-byte under
`evidence/gptw_response_fibre_v1_1_0/`. It retains the formal v1-r1 failure,
the same-seed v1-r2a precision repair, the independent v1-r3 new-seed
adaptive-value confirmation, the v2 current-versus-source-fixed extension,
and the v3 prospective disjoint natural-English audit. The bundled
`SHA256SUMS` and `verify_snapshot.py` jointly verify file identity, cohort
counts, decisions and V3 node-level gates.

The scope remains the rank-2 LoRA-B subspace of the final two GPT-2 blocks.
No claim is made for full-model GPT-2, arbitrary prompts, semantic invariance,
capacity-weighted CNER, or arbitrary-path/global variational optimality.
