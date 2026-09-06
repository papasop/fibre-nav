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
The review-branch wrapper pins the executable source snapshot at commit
`00500e1322be67c9774d44c15e44e598d6ec9039`; after a release tag is created,
the Colab-facing link should be moved to that immutable tag.

Scientific role: same-cohort external reproduction of the prospectively frozen
8-seed/24-node GPT-2 native-LoRA-B Pareto audit, not an independent new-seed
confirmation. No frozen result, gate, seed count or decision is changed.

## GPT-2 LoRA-B Pareto R2 strict-control confirmation

This release imports
`neural_fibre_geometry_gpt2_lora_pareto_r2_github_update.zip` as an independent
prospective strict-control upgrade under
`evidence/low_response_pareto_v1/gpt2_lora_b_r2_strict/`. ZIP SHA-256:
`9b250a836f911ded7df1d7a06cab9451261a591819a3da1990a58f9ea44b320e`.

R2 does not overwrite, repair or retroactively strengthen R1. It adds a
calibration-selected best-of-16 equal-norm random direction inside the same
current response kernel. The frozen decision is
`GPT2_LORA_LOW_RESPONSE_PARETO_STRICT_CONFIRMED`: 8/8 prospective GPU seeds
completed, 6/8 seeds passed the complete frozen control family, 7/8 seed-level
same-kernel contrasts were positive, and 20/24 noninitial nodes were positive.

The imported verifier `verify_gpt2_lora_pareto_r2.py` checks file hashes,
protocol equality, cohort counts, node counts, bootstrap interval positivity
and the exact two-sided sign-flip p-value before commit.

## GPT-2 LoRA-B R2 predictive and causal boundary overlay

This release imports
`neural_fibre_geometry_r2_pc1_pc4_github_overlay_r1.zip` as an add-only overlay.
ZIP SHA-256:
`18c7da256e1c4e89b496e814f5d0a23dde4fbd924e4d97b09f746d30d706a334`.

The overlay adds a root-level R2 strict-control evidence layout under
`evidence/gpt2_lora_r2_strict/` and predictive/causal boundary audits under
`evidence/predictive_causal_boundaries/`. PC1 reports
`PREDICTIVE_RESPONSE_FIBRE_CAPACITY_NOT_SUPPORTED`, PC3 reports
`DYNAMIC_RESPONSE_FIBRE_CAPACITY_PREDICTIVE_STATE_NOT_SUPPORTED`, and PC4
reports `CAUSAL_RESPONSE_FIBRE_CAPACITY_NOT_SUPPORTED`. PC2 source and protocol
are archived, but its authoritative result artifact is absent and remains
required before a repository-backed PC2 claim.

No historical R1 or low-response R2 raw result is deleted or relabelled. The
boundary overlay narrows interpretation of the R2 confirmation: it remains a
counterfactual finite-budget held-out frontier at recorded nodes, not a
predictive or causal law for ordinary optimizer futures.

## Cached intrinsic Picard fine-tuning v0.2.6

This release imports `picard_finetune_github_v0_2_6.zip` as an independent
supported side-branch evidence snapshot under
`evidence/picard_finetune_v0_2_6/`. ZIP SHA-256:
`9c88601620c11f1cdcf09d8607400a2a0df0d25803a142122fe799757f072899`.

The frozen protocol is
`GEOMETRIC_INTRINSIC_PICARD_FINETUNE_V0_2_6_REPEATED_TIMING_CONFIRMATORY`.
The machine-readable status is
`PICARD_V0_2_6_REPEATED_TIMING_DUAL_10PCT_SPEEDUP_SUPPORTED`. Five new
evaluation seeds passed every frozen timing, endpoint, accuracy and
float64-response-leakage gate. Median time-to-equal-loss speedup was 24.46%,
median fixed-budget speedup was 16.70%, median steps-to-target reduction was
9.09%, median endpoint loss delta was -0.000099 and certified response leakage
was 8.23e-16.

This import preserves the original returned results archive
`evidence/v0_2_6/picard_v0_2_6_results.zip` inside the snapshot and adds a
repository-level verifier. Its scope is frozen-feature ResNet-18/CIFAR-10 in a
20-dimensional float64-certified intrinsic response kernel; it is not
end-to-end fine-tuning, GPT-2/LoRA evidence, a universal optimizer comparison
or a global Picard-flow theorem.

## GPT-2 LoRA Picard frozen confirmation v1.6.0

This release imports the GPT-2 LoRA Picard r1-r5 evidence chain while preserving
the earlier CIFAR Picard v0.2.6 snapshot as secondary support.

Imported source/result ZIP SHA-256 values:

- `picard_gpt2_lora_quick_v0_2_6_r1.zip`:
  `b8d42803560621e6801458b12f5f016bf233ce26ec7a55778b05ae708dac33b1`
- `picard_gpt2_lora_quick_v0_2_6_r1_results.zip`:
  `56e86103f0e3527db9a72a1ef03cf8ca1f48e3081bdba8d1c47d61d65ac73e4e`
- `picard_gpt2_lora_quick_v0_2_6_r2.zip`:
  `85f4cbbe2ecf93c0b279edbac26e53099262fe35f73e1ef122e1db9d6ccb7100`
- `picard_gpt2_lora_quick_v0_2_6_r2_results.zip`:
  `1f2728501d7582d6e562f3621cad0b36212e7853966f5b61cc0efd3a3adaad3f`
- `picard_gpt2_lora_quick_v0_2_6_r3.zip`:
  `e000a2217ae8baa3130a44955d5cc50bd434daba2c129a118dec1c547cb4caeb`
- `picard_gpt2_lora_quick_v0_2_6_r3_results.zip`:
  `79f2f638a6b4834847513eca016a3ee78d9106d7fb17f6451d2abf0680a01a73`
- `picard_gpt2_lora_confirm_v0_2_6_r4.zip`:
  `8b1aae50ba28e981d49f4a2c40eac60dd193fd88139565f597ec8afc5ca8a105`
- `picard_gpt2_lora_confirm_v0_2_6_r4_results.zip`:
  `d1e18f8cc12c60bc13d1a72b4c56a55efff2a9babbb7b8bf19bcb3dc0a108547`
- `picard_gpt2_lora_confirm_v0_2_6_r5.zip`:
  `0ed7bc923ffae5fff596b5ceb90b215834281cbb3dd528dd8c75105adaa67ba1`
- `picard_gpt2_lora_confirm_v0_2_6_r5_results.zip`:
  `c86e3ab46e3dc65893b6d5cbf6dda8acc6b78ad589aa4c8cce8029b580d8e0b1`

R4 is the repository-backed frozen five-new-seed confirmation:
`GPT2_LORA_PICARD_V0_2_6_R4_FROZEN_CONFIRMATORY_SUPPORTED`, with 5/5 positive
seeds. R5 is a ten-step target-detection resolution audit on the same seed
cohort: `GPT2_LORA_PICARD_V0_2_6_R5_TEN_STEP_RESOLUTION_SUPPORTED`, with
37.46% median time-to-equal-loss speedup, 5/5 positive seeds and 1.17%
fixed-budget compute-time advantage retained as diagnostic only.

R1-R3 are preserved under `evidence/developmental/picard_gpt2_lora/` as the
device, dtype, direct-Fisher and metric-exponent development history. R4 is
archived under `evidence/confirmed/picard_gpt2_lora_r4/`; R5 is archived under
`evidence/audits/picard_gpt2_lora_r5_ten_step/` and is the default external
reproduction route. The claim is task-specific GPT-2 small LoRA evidence, not
full-model GPT-2 fine-tuning, semantic-invariance evidence, downstream-task
transfer, universal optimizer superiority or a global Picard-flow theorem.

## GPT-2 LoRA global response-budget R10/R11

This import adds `github_r10_r11_evidence_bundle.zip` as an independent
current-versus-source response-kernel global-budget branch under
`evidence/gpt2_lora_global_response_budget_r10_r11/`. The observed local ZIP
SHA-256 was
`d8a3e0c8a8565f69fec098b6aaf731ccf142639b6440f07e47cfe7ad94629d99`; no
external expected SHA-256 was supplied in the user request.

R10 is development:
`R10_GLOBAL_BUDGET_PARETO_DIAGNOSTIC_COMPLETE`, using intentionally reused
R7-R9 development seeds. R11 is the untouched-seed confirmation:
`R11_CURRENT_KERNEL_GLOBAL_BUDGET_PARETO_CONFIRMED`. It uses five untouched
seeds and four frozen global response budgets. At budgets `5e-5`, `1e-4`,
`2e-4` and `5e-4`, the median source-minus-current validation-loss gaps were
0.003762603, 0.003669620, 0.003578544 and 0.003186703, respectively, with 5/5
current-positive seeds at each budget. The five seeds are the independent
experimental units; the 20 seed-by-budget contrasts are correlated.

The scope is a deterministic compact randomly initialized GPT-2 byte language
model and shared 24-dimensional rank-4 LoRA chart. This evidence is separate
from the r4/r5 Picard-vs-AdamW timing branch and does not establish AdamW
superiority, training speed, pretrained GPT-2 fine-tuning, semantic transfer,
inverse-Fisher superiority, universal optimizer ordering or a global Picard
theorem.

## Pretrained GPT-2 dynamic response-kernel R12/R13

This import adds `pretrained_gpt2_dynamic_kernel_r12_r13_github_bundle.zip`
under `evidence/pretrained_gpt2_dynamic_kernel_r12_r13/`. The expected and
observed ZIP SHA-256 was
`cd4b9aadbc7b4b2738a9fa82520b5fdb626e9545cec3d0632f463e9da318f1ad`.

The snapshot preserves R12a, R12a-r1, R12a-r2 and R12b as development and
repair stages, and R13 as the frozen five-untouched-seed confirmation:
`R13_PRETRAINED_GPT2_CURRENT_KERNEL_BUDGETED_DUAL_ADVANTAGE_CONFIRMED`.
The confirmed setting is `openai-community/gpt2`, SHA-256-bound Tiny
Shakespeare data, two adapted final `c_attn` modules and a shared
24-dimensional rank-4 LoRA chart. R13 used seeds `33211`, `33217`, `33229`,
`33241` and `33253` and budgets `2e-5` and `5e-5`. Current-kernel identity
updates beat both source-frozen updates and response-budgeted AdamW in all five
seeds at both budgets. Median budgeted-AdamW-minus-current validation-loss gaps
were 0.001384894 and 0.001366377; median source-minus-current gaps were
0.001331011 and 0.001289566.

The original returned result ZIPs are preserved for R12a, R12a-r1, R12a-r2 and
R13. The original R12b result ZIP was not supplied; its results are disclosed
as mechanically recovered from a complete returned Colab log embedded in the
archive. This provenance distinction remains part of the imported record.

Unconstrained AdamW reached lower validation loss but exceeded the response
budgets by hundreds of times. It is retained as a scope diagnostic, not as a
matched-budget control. This evidence does not establish superiority to
unconstrained AdamW, ordinary full-model GPT-2 training, semantic transfer,
downstream-task transfer, inverse-Fisher superiority, lower universal per-step
complexity, universal optimizer superiority or a global Picard theorem.

## Pythia-160M metric-constrained Onsager R17d

This import adds `fibre_nav_r17d_github_overlay.zip` as an add-only evidence
overlay. The expected and observed ZIP SHA-256 was
`b01d0dd56353c804030aab8844d82c15dc69e6ccc715017d170a2180ba5c82eb`.

The confirmed archive is
`evidence/confirmed/pythia160m_metric_onsager_r17d/`, with authoritative status
`R17D_METRIC_CONSTRAINED_ONSAGER_CONFIRMED`. R17d tests a pretrained
Pythia-160M prompted GLUE/SST-2 learning target, a disjoint frozen AG News
response map, one 32-dimensional LoRA chart and one global response-budget
ceiling. The frozen metric-constrained Onsager arm passed the aggregate gate on
five untouched seeds: four of five seeds passed the complete frozen seedwise
gate, all numerical, response-budget, rank, KKT and finite-value gates passed,
median AdamW-minus-metric-Onsager validation loss was +0.00135338, median
source-frozen-minus-current validation loss was +0.01599610, and median
metric-Onsager-minus-AdamW accuracy was -0.390625 percentage points, inside the
frozen -0.5-point noninferiority margin.

The development chain is archived under
`evidence/developmental/pythia160m_metric_onsager_r17_chain/`: R17a is
Euclidean constrained-Onsager diagnosis, R17b is adaptive-metric diagnosis and
R17c is same-seed budget calibration. R17c is not confirmation. R16b is
archived under `evidence/developmental/pythia160m_response_action_r16b/` as the
response-occupancy-only action negative boundary. R17c and R16b disclose that
the available Colab logs are preserved verbatim and their JSON summaries are
mechanical recoveries from those logs.

This evidence does not establish a continuous-action theorem, universal
optimizer ordering, Principle-R theorem, physical law, equal-realized-budget
result or claim that every seed beats projected AdamW. It does not replace the
R13 pretrained GPT-2 dynamic-kernel confirmation; it adds a separate
model-specific variational implementation bridge.

## Fibre-Qwen v0.0.1-development overlay

This import adds `fibre-qwen-v0.0.1-development-github.zip` as a development
side overlay under `fibre-qwen/`. The expected and observed ZIP SHA-256 was
`4dd1670c131de44b063906493a4baf0fe6921655d94beeaeb065be640b3c3e8f`.
The package manifest was verified before import; the in-repository
`fibre-qwen/MANIFEST.sha256` is regenerated for the final `fibre-qwen/` path.

The overlay preserves R20-R21 protocols, short conversation-derived preference
bootstraps, frozen evaluation summaries, targeted rule profiles and a
deterministic R21b-r2 router/oracle diagnostic protocol. It contains no model
weights, adapter checkpoints, private raw dialogue, credentials, caches,
`__pycache__` directories or compiled Python bytecode.

The current Fibre-Qwen state is fail-closed. R21b-r1 reports generation health
9/10 and improved declared-answer margins on 8/10 reused items, but semantic
success remains 5/10 below the frozen semantic gate. R21b-r2 is protocol only;
no R21b-r2 result is claimed. This overlay is not moving-response-fibre
optimizer evidence, broad personalization evidence, safety certification,
continual learning or a trained-model release.


## Fibre-Qwen R21b-r2 through R21c-r1 update

The development overlay was updated from parent commit `5700a527ea60639178291f0b2b7ba2a6abb61adb` using archive SHA-256 `ac0e4d7a0017093e83ad96c618d8002d3c0583cf9991338533c569a1111baffd`. R21b-r2 achieved exact deterministic routing, but its original result ZIP was unavailable at packaging time; only a transcript-reconstructed summary is included and explicitly marked, with no fabricated per-item record. R21c preserves the exact 40-record Qwen3-0.6B/Qwen3-8B comparison. R21c-r1 mechanically invalidates the v1 evaluator and reports post-hoc v2 rescoring. The 8B 7/10 score is a development nomination only, not untouched confirmation, personalization, safety certification, weight learning or deployment readiness.

## Moving Fibre Intelligence L3 writer v1.7.8

This import adds `moving_fibre_intelligence_l3_writer_v1_7_8_github.zip` under
`evidence/l3_writer_v1_7_8/`. The expected and observed ZIP SHA-256 was
`c6568d5f5abc3e604c03529a05ac38d0efb1f503df0831270bcc12e3a75976a8`.
The package `SHA256SUMS.txt` was verified before import, and the in-repository
`evidence/l3_writer_v1_7_8/SHA256SUMS.txt` records the final imported layout.

The archived status is `DEVELOPMENT_NOT_CONFIRMATION`. It is a single frozen
GPT-2 seed, `81902`, for protocol
`MOVING_FIBRE_INTELLIGENCE_L3_MIXED_SLOT_MARGIN_V1_7_8`. The run completed one
seed and reports `all_seeds_pass: true`, `all_seeds_tertiary_pass: true`,
configuration SHA-256
`76619bb7ed40ee3bee14bdc0f8ae3f768431486474e7188f973909715649681b`, PyTorch
`2.11.0+cu128`, Transformers `4.57.6`, and elapsed time 2592.3215227127075
seconds.

The supported scope is exact memory access, held-out linguistic access for
previously written concepts, two-slot compositional codes, cyclic concept-code
reassignment, response compliance and endpoint-KL compliance. This archive
does not support whole-concept category generalization, geodesic semantic
distance, curvature, parallel transport, a certified fibre connection, a
global section, model-family generality or multi-seed confirmation. It does
not include v1.8.0 or v1.8.1 evidence and is not a Hugging Face model release.

## Moving Fibre Intelligence L3 shared-category writer v1.8.1

This import adds `fibre_nav_mfi_l3_v1_8_1_github_overlay.zip` under
`evidence/l3_category_v1_8_1/`. The observed local ZIP SHA-256 was
`c7df2cfa7d8dbec62ea9d23d9165bbd44db7e235fd28369823b267ebd62d13ab`; no
separate expected SHA-256 was supplied in the user request. The package
`SHA256SUMS.txt` was verified before import, and the in-repository
`evidence/l3_category_v1_8_1/SHA256SUMS.txt` records the final imported layout,
including the archived import note and repository audit.

The archived status is `DEVELOPMENT_NOT_CONFIRMATION`. It is a single frozen
GPT-2 seed, `82001`, for protocol
`MOVING_FIBRE_INTELLIGENCE_L3_SHARED_CATEGORY_WRITER_V1_8_1`. The run completed
one seed and reports `all_seeds_pass: false`, `all_seeds_tertiary_pass: false`,
configuration SHA-256
`b91ff3abae25dd7f21dc1dcfe68474db90d5c788d2d42d9e480b9bf48a24604f`, PyTorch
`2.11.0+cu128`, Transformers `4.57.6`, and elapsed time 3998.7832641601562
seconds.

The supported scope is a partial category-conditioned writer result: 16/16
shared-router training accuracy, 7/8 whole-concept held-out routing, 24/24
eligible category moves, category-current improvement over shuffled-category
and no-graph controls, and exact/training-expression access for the declared
concepts. The failed or unsupported gates are positive held-out category margin
for every concept, within-category distances smaller than between-category
distances, improvement over the pair-only control, universal held-out
linguistic access after category movement, category-geometry confirmation,
model-family generality and multi-seed confirmation.

## Moving Fibre Intelligence L3 v1.9.0/v1.9.1 category geometry

This import adds `moving_fibre_intelligence_l3_v1_9_1_github.zip` under
`evidence/l3_category_v1_9_1/`. The expected and observed ZIP SHA-256 was
`acbf0c688e36abab047236b7c2497797540fd775784f5c4273d31b7dade66bbf`.
The package `SHA256SUMS.txt` was verified before import, and the in-repository
`evidence/l3_category_v1_9_1/SHA256SUMS.txt` records the final imported layout.

The archive contains two distinct states. v1.9.0 is a single-seed GPT-2
development result in seed `82001` for protocol
`MOVING_FIBRE_INTELLIGENCE_L3_PROTOTYPE_METRIC_REPAIR_V1_9_0`. It reports
`all_seeds_pass: true`, `all_seeds_tertiary_pass: true`, configuration
SHA-256 `d81e7006a39d7c0e47898e2260423f97e4e5517a5ae56bda788789a226e4e90a`,
PyTorch `2.11.0+cu128`, Transformers `4.57.6`, and elapsed time
4498.721615552902 seconds. The README records the headline v1.9.0 category
gates: 8/8 held-out whole-concept routes, minimum held-out distance margin
`+1.9900`, within/between category-coordinate ratio `0.0301`, and failed
control gates for shuffled-category, pair-only and no-graph alternatives.

v1.9.1 is not a result import. It freezes source, configuration, Colab
launcher and release assets for a prospective confirmation on fresh seed
`82101` and a new balanced held-out split. Its status is
`PROSPECTIVE_CONFIRMATION_PENDING`; no v1.9.1 positive or negative scientific
decision is repository-backed by this import. L4 cross-fibre transport remains
untested.

## Moving Fibre Intelligence L3 v1.9.6.0 single-seed confirmation

This import adds `moving_fibre_intelligence_l3_v1_9_6_0_github_archive.zip`
under `evidence/l3_category_v1_9_6_0/`. The expected and observed ZIP SHA-256
was `26ba5594381a2b0591f7446df6ec59eee71f7fcce8a7d9e09ecd53c453a1e484`.
The package `SHA256SUMS.txt` was verified before import, and the in-repository
`evidence/l3_category_v1_9_6_0/SHA256SUMS.txt` records the final imported
layout.

The archived status is a prospectively frozen single-seed L3 confirmation in
the tested GPT-2 LoRA setting, not multi-seed confirmation. The run used writer
seed `82601`, router split seed `196001`, four new bridge instances from the
existing 80-concept cohort, and protocol
`MOVING_FIBRE_INTELLIGENCE_L3_PROSPECTIVE_CONFIRMATION_V1_9_6_0`. It reports
`all_gates_pass: true`, configuration SHA-256
`9f06fcf505df7b4a0ccbfea59d6e866eb3f8c29929227fcc5af55b332f1c829e`, PyTorch
`2.11.0+cu128`, Transformers `4.57.6`, and elapsed time 3189.910411119461
seconds.

The router preflight passed every declared routing gate with 79/80
outer-held-out accuracy, 93.75% worst-fold accuracy and 95% minimum class
recall. The integrated bridge routed `dog`, `tokyo`, `japan` and `apple`
correctly with positive margins, and 16/16 eligible rank-8 writer endpoints
passed the declared exact, training-expression, held-out-expression,
mixed-address, response-budget and endpoint-KL gates. This import does not
support multi-seed statistical stability, an entirely new external concept
cohort, cross-model generality or L4 cross-fibre transport.
