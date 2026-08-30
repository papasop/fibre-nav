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

## v1.5.0 ResNet mechanism and SI closure

This release imports the original v4.1b and v4.2d executable packages and
result ZIPs byte-for-byte. It also expands their 32 per-seed JSON records and
adds an independent reconstruction script that ignores the aggregate values in
`report.json`. The reconstructed values agree with Sections 5.3 and 6 of the
paper at the displayed precision.

v4.1b used new seeds 68726--68741. All twelve paths for each seed were generated
and frozen from development data before the first confirmation access. All
16/16 seeds passed every geometric and held-out gate. Its parameter domain is a
trainable terminal residual adapter plus classifier over frozen ResNet-18
features, not the complete backbone.

v4.2d used new seeds 76742--76757 after a float32/TF32 development attempt
failed the frozen JVP identity certification. The confirmation used float64
response geometry with TF32 disabled, trainable layer4 plus classifier, and an
eight-dimensional matrix-free chart. All 16/16 seeds passed the frozen 12/16
joint gate. The failed precision stage is described as failure history and is
not counted as positive evidence.

Neither result establishes a complete high-dimensional kernel bundle,
universal scaling, ResNet F16 ordering, or local/global variational optimality.

## v1.5.1 GPTW-V3 external reproduction interface

This release adds only an execution and provenance wrapper. The wrapper pins
the already confirmed GPTW source at tag
`v1.4.0-gptw-natural-text-confirmed`, commit
`236f646c472018a7e38be11fd658519763bc2346`; verifies its byte-level
`SHA256SUMS` and semantic cohort checks; and runs the unmodified V3 program.

The resulting archive records the environment, GPU, pinned source identity,
decision, cohort counts and result checksum. No seed, prompt, gate, model,
parameter domain or scientific decision rule changes in v1.5.1. Consequently
this interface supports same-cohort external reproduction, not an independent
new-seed or new-prompt confirmation.

## GPTW claim calibration and v4.4-r1 staging

This documentation correction calibrates the GPTW v2/v3
current-versus-source-fixed interpretation. The 2-versus-1 exponent split is
analytically forced by the arm definitions and retained as a
numerical-correctness check. The repository-facing GPTW evidence is expressed
in terms of finite-radius cost ratios, principal/current-source angular
separation where recorded, active-J residual amplification, numerical-precision
path-response error and prospective seed/node replication.

This correction does not change any GPTW engine, per-seed result JSON, report,
gate or scientific decision. The v2 codeword branch remains SUPPORTED, the v1-r3
adaptive-value result remains confirmed, and v3 natural text supports only that
the current-versus-source-fixed separation is not confined to the original
codeword prompts.

This revision also stages ResNet v4.4-r1 as a developmental frozen-code and
protocol record for an ordinary AdamW current-versus-source response-geometry
audit. The authoritative recovered results ZIP is intentionally deferred; no
positive or negative v4.4-r1 decision is repository-backed by this staging
record.

## Low-response Pareto v1

This release imports `neural_fibre_geometry_low_response_pareto_github_ingest_v1.zip`
as a root-relative evidence snapshot. ZIP SHA-256:
`58e122b00444aca8ea161223437dcfcc0d174bb94d52206fe17b41c37a21d8e6`.

The snapshot contains two prospective CPU audits of held-out low-response
utility frontiers. Reduced ResNet v4.6 reports
`LOW_RESPONSE_PARETO_ADVANTAGE_SUPPORTED` in 7/8 seeds. GPT-2 native LoRA-B
reports `GPT2_LORA_LOW_RESPONSE_PARETO_CONFIRMED` in 8/8 seeds and 24/24
noninitial nodes. These are finite-chart, finite-budget, frozen-alpha-grid
claims. The successful instantaneous-kernel arm is a counterfactual projection,
so the import does not establish that ordinary AdamW or SGD naturally follows a
response fibre.

The imported paper PDF is a manuscript snapshot that distinguishes this
24-node Pareto audit from the separately frozen 32-node GPTW-v3
correction-cost audit.

## Main-branch README and manuscript hierarchy repair

This documentation revision retitles the repository-facing paper snapshot as
`Moving Response Fibres: A Geometric View of Behaviour and Learning` and
reorganizes the root README around one moving-response-fibre mechanism, two
main results, limited GPTW correction-cost confirmation and the restricted F16
failure boundary.

The current paper-facing PDF is
`paper/Moving_Response_Fibres_A_Geometric_View_of_Behaviour_and_Learning.pdf`.
The previous reviewed snapshot is preserved without deletion at
`paper/archive/Moving_Response_Fibres_v5_unified_hierarchy.pdf`.

No experimental result, protocol, seed count, gate, raw archive or scientific
decision changed in this hierarchy and manuscript-snapshot repair.

## GPT-2 LoRA-B Pareto reproduction repair

This revision restores the parent engine required by the archived GPT-2
native-LoRA-B eight-seed wrapper:
`evidence/low_response_pareto_v1/gpt2_lora_b_v1/gptw_lora_low_response_pareto_cpu.py`.
The file is extracted from the original frozen input ZIP
`gptw_gpt2_lora_pareto_cpu_8seed_v1_r1.zip`; it is not a reconstructed or
functionally similar replacement.

Authoritative repair hashes:

- original eight-seed ZIP: `ce0fe0ebc201478b8cac62cc8b067272bbf9ecbb851a87d5159c1ec9627eb08e`
- restored parent engine: `531d08d128d31c8de1fac61a39fb2915c2de8b09561efc383430d444bab3b773`

It also adds a zero-upload external reproduction wrapper under
`external_tests/gpt2_lora_pareto/`. The wrapper verifies the archived snapshot,
checks the parent engine and eight-seed wrapper, and can run archive-only,
smoke-test or full same-cohort reproduction modes.

Scientific role: same-cohort external reproduction of the prospectively frozen
8-seed/24-node GPT-2 native-LoRA-B Pareto audit, not an independent new-seed
confirmation. No frozen result, gate, seed count or decision is changed.
