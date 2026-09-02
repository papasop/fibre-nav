# Moving Response Fibres

## A Geometric View of Behaviour and Learning

Current main-branch evidence snapshot for the associated paper by Y. Y. N. Li.
This repository archives frozen programs, protocols, results, failed stages
and verification records.

Here, *behaviour* means the value of a prospectively declared finite response
map \(R(\theta)\), not a model's complete input-output function. *Learning*
means improvement on a separately declared adaptation objective, evaluated on
held-out data where specified. A response fibre therefore collects internal
realizations that agree on the declared behaviour but may differ in their
capacity for subsequent learning.

The archive is organized around one featured GPT-2 LoRA Picard result, one
central response-fibre mechanism, two moving-fibre evidence branches, limited
cross-architecture confirmations and strict failure boundaries. Developmental,
failed and excluded stages are preserved rather than relabelled after
inspection.

## Featured result: GPT-2 LoRA Picard vs AdamW

The headline result is a frozen GPT-2 small LoRA comparison between a cached
intrinsic Picard update and AdamW on SHA-256-bound Tiny Shakespeare bytes.
Both arms start from the same 50-step AdamW warm start, which is excluded from
comparison timing. The trainable domain is rank-4 LoRA on `c_attn`.

| Primary metric | Result |
| --- | ---: |
| Median time-to-equal-loss training speedup | **37.46%** |
| Positive seeds | **5/5** |
| Fixed 600-step compute-time advantage | **1.17%, diagnostic only** |

r4 is the formal frozen five-new-seed confirmation:
`GPT2_LORA_PICARD_V0_2_6_R4_FROZEN_CONFIRMATORY_SUPPORTED`. r5 is the
ten-step target-detection resolution audit:
`GPT2_LORA_PICARD_V0_2_6_R5_TEN_STEP_RESOLUTION_SUPPORTED`. r5 reuses the same
five seeds as r4, so it is not a second independent confirmation; it is the
default external reproduction path because the target crossing is resolved at
10-step granularity rather than 50-step granularity.

The result is task-specific GPT-2 LoRA evidence. It does not establish a
universal optimizer ordering, ordinary full-model GPT-2 fine-tuning, semantic
invariance, downstream-task transfer, or a global Picard-flow theorem.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/papasop/fibre-nav/blob/main/external_tests/picard_gpt2_lora/GPT2_LORA_PICARD_R5_ONE_CLICK.ipynb)

```bash
python verify_picard_gpt2_lora_v1_6.py
python external_tests/picard_gpt2_lora/COLAB_ONE_CLICK.py --source-root . --verify-only
```

## Central mechanism: behaviour does not determine learning freedom

For a prospectively frozen response map \(R\), local response-preserving
freedom is represented by

\[
V_\theta = \ker DR(\theta).
\]

The central claim is not merely that this kernel is large. It is that the
relevant response-preserving distribution changes with model state, and that
using its instantaneous rather than stale geometry has finite geometric and
learning consequences.

The repository evidence does not show that every kernel direction is useful,
that complete model behaviour is preserved, or that ordinary optimizers
naturally follow the response fibre.

## Strongest response-fibre frontier result

At recorded AdamW nodes, the instantaneous response kernel defines a superior
held-out learning frontier under the same prospectively frozen finite response
budgets.

In the strict GPT-2 native-LoRA-B R2 audit:

- 8/8 prospectively frozen seeds passed;
- 6/8 seeds passed the complete frozen control family;
- the current-minus-random-current-kernel seed contrast was positive in 7/8
  seeds and 20/24 noninitial nodes;
- the bootstrap 95% confidence interval for the mean seed contrast was
  [0.0296, 0.1283] with exact two-sided sign-flip \(p=0.03125\).

Candidate scales were selected on calibration prompts and evaluated once on
disjoint held-out templates. The successful arm is a counterfactual projection
at recorded optimizer nodes; the audit does not show that AdamW naturally
discovers or follows the response fibre.

This R2 strict-control upgrade supports task-specific low-response Pareto
advantage beyond generic current-kernel membership. Historical R1 files and
decisions remain unchanged.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/papasop/fibre-nav/blob/v1.6.0/external_tests/gpt2_lora_pareto/GPT2_LORA_PARETO_EXTERNAL_ONE_CLICK.ipynb)

The Colab link targets the immutable `v1.6.0` tag. The launcher checks out and
verifies fixed source commit `00500e1322be67c9774d44c15e44e598d6ec9039`.

[Inspect protocol and archived results](evidence/low_response_pareto_v1/gpt2_lora_b_r2_strict/)

## Dynamic response-kernel global-budget confirmation

R10/R11 add a separate current-versus-source response-kernel Pareto audit under
four frozen global response balls in a compact randomly initialized GPT-2 byte
language model. R10 is same-seed development; R11 freezes the configuration
and evaluates five untouched seeds.

R11 confirmed the current-kernel arm over the source-frozen arm in all four
budgets, with 5/5 positive seeds at each budget. Median source-minus-current
validation-loss gaps were 0.003762603, 0.003669620, 0.003578544 and
0.003186703 for budgets `5e-5`, `1e-4`, `2e-4` and `5e-4`, respectively.
The five seeds are the independent units; the 20 seed-by-budget contrasts are
correlated. This is not an AdamW, speed, pretrained GPT-2, semantic-transfer,
universal-optimizer or global-Picard claim.

[Inspect R10/R11 evidence](evidence/gpt2_lora_global_response_budget_r10_r11/)

## Pretrained GPT-2 dynamic-kernel confirmation

R12/R13 extend the dynamic response-kernel global-budget test from a compact
random byte model to `openai-community/gpt2` with a shared 24-dimensional
rank-4 LoRA chart in the final two `c_attn` modules and SHA-256-bound Tiny
Shakespeare data. R12 is the development chain; R13 is the frozen five-seed
confirmation under two declared global response balls.

R13 confirmed that current-response-kernel identity updates achieved lower
validation loss than both source-frozen updates and response-budgeted AdamW in
all five untouched seeds at both budgets. Median budgeted-AdamW-minus-current
loss gaps were 0.001384894 and 0.001366377 at budgets `2e-5` and `5e-5`;
median source-minus-current gaps were 0.001331011 and 0.001289566. The five
seeds are the independent units; the ten seed-by-budget contrasts are
correlated. Unconstrained AdamW reached lower loss but exceeded the response
budgets by hundreds of times, so it is retained as a scope diagnostic rather
than part of the matched-budget confirmation gate.

[Inspect R12/R13 evidence](evidence/pretrained_gpt2_dynamic_kernel_r12_r13/)

## Pythia-160M constrained Onsager confirmation

R17d tests whether the instantaneous response kernel can enter a prospectively
declared incremental variational update, rather than only a counterfactual
finite-budget direction audit. The learning target is prompted GLUE/SST-2; the
independently declared response map comprises four AG News topic-margin
coordinates on frozen, disjoint inputs.

For Adam first- and second-moment statistics \(\hat m_k,\hat v_k\), define
\(M_k=\operatorname{diag}(\sqrt{\hat v_k}+\epsilon)\). The frozen candidate is
the exact metric-constrained Onsager step:

\[
\min_{DR(\theta_k)\delta=0}
\left[\hat m_k^T\delta+\frac{1}{2\eta_k}\delta^TM_k\delta\right].
\]

R17c intentionally reused its development seed to select multiplier 1.15 under
a predeclared response-budget-utilization rule. R17d then evaluated that
configuration once on five untouched seeds against current-kernel projected
AdamW and source-frozen metric Onsager under the same global response-budget
ceiling.

The R17d authoritative status is
`R17D_METRIC_CONSTRAINED_ONSAGER_CONFIRMED`: 4/5 untouched seeds passed the
complete frozen seedwise gate, with all numerical, response-budget, rank, KKT
and finite-value gates passing. Median AdamW-minus-metric-Onsager validation
loss was +0.00135338, median source-frozen-minus-current validation loss was
+0.01599610, and median metric-Onsager-minus-AdamW accuracy was -0.390625
percentage points, inside the frozen -0.5-point noninferiority margin.

This is a model- and protocol-specific confirmation within pretrained
Pythia-160M, one 32-dimensional LoRA chart, one learning task, one independent
response map and one global response budget. It is not a continuous-action
theorem, universal optimizer ordering, Principle-R theorem or physical law.
One seed strongly favored projected AdamW, and the comparison is under a
shared budget ceiling rather than equal realized budget consumption in every
seed. R16b is retained as the response-occupancy-only action negative boundary.

[Inspect R17d evidence](evidence/confirmed/pythia160m_metric_onsager_r17d/)

## Fibre-Qwen development overlay

`fibre-qwen/` archives an auditable v0.0.1-development side project for a
research-oriented Qwen assistant. It contains R20-R21 protocols, short
conversation-derived preference bootstraps, frozen evaluation summaries,
targeted rule profiles and a deterministic R21b-r2 router/oracle diagnostic
protocol. It does not include model weights, adapter checkpoints, private raw
dialogue, credentials, caches or `__pycache__`.

The current Qwen state is fail-closed. R21b-r1 restored generation health to
9/10 and improved declared-answer margins on 8/10 reused items, but semantic
success remained 5/10, below the frozen semantic gate. R21b-r2 is protocol
only in this archive; no R21b-r2 result is claimed. This branch is not
moving-response-fibre optimizer evidence, broad personalization evidence,
safety certification, continual learning or a trained-model release.

[Inspect Fibre-Qwen overlay](fibre-qwen/)

## Functional premise: selected fibre directions can carry adaptive value

ResNet-18/CIFAR-10 v4.0c-r1 prospectively tested the functional premise in the
complete 5,130-parameter final classifier on top of a frozen ImageNet-pretrained
ResNet-18 representation. Sixteen new classifier-training seeds were evaluated
at 4, 16 and 32 anchors. Every seed passed every frozen response-preservation,
effect-size and control gate at all three anchor counts. The learned tangent
beat sign reversal, a shuffled-target tangent and the best of 32 matched random
kernel tangents in all 48 seed-anchor settings.

At the primary 16-anchor setting, the learned tangent's median confirmation-loss
change was -0.01842, compared with +0.02307 for sign reversal, -0.00099 for the
shuffled-target tangent and -0.00072 for the best random tangent. The maximum
dimensionless kernel residual was 2.28e-13 and maximum finite centered-logit
leakage was 2.98e-13. This confirms selected tangent value in a frozen
pretrained representation; it does not update the ResNet backbone or provide
cross-model confirmation of moving-kernel transport, cost scaling or
Moving-Fibre F16 ordering.

This is a fixed-chart functional premise. It is distinct from the moving-fibre
geometry audits and from the real-node low-response Pareto frontier below.

## Main result I: finite consequences of moving response geometry

Current-kernel transport and source-fixed replay differ in finite
realizability cost, and minute drift between their parameter directions can be
strongly amplified in active response space. Realizability-cost scaling and
transverse amplification are treated here as two measured consequences of one
moving-geometric result, not as separate theoretical centres.

Moving-Fibre v3.1e compared an online eight-dimensional chart reprojected into
the current response kernel with replay in a stale source-fixed chart. Of 16
new seeds, 15 were fully comparable and all 15 passed every frozen substantive
gate. Median scaling statistics were

\[
\alpha_{\rm moving}=0.684,\qquad
\alpha_{\rm fixed}=0.00715,
\]

with smallest-radius fixed/moving retraction-Fisher cost ratio 21.30 and
tangent-residual ratio 21.47. This confirms finite-radius realizability-cost
scaling in the frozen CNN--MNIST construction; it is not a continuum
fibre-bundle theorem.

ResNet v4.1b extends the finite-radius scaling audit to a frozen
ImageNet-pretrained ResNet-18 backbone with a trainable terminal residual
adapter and classifier. All 16/16 seeds passed every geometric and held-out
gate; recomputed medians give slope gap 0.5751 and smallest-radius cost ratio
2.997.

ResNet v4.2d tests transverse response amplification with trainable layer4 plus
classifier, an eight-dimensional matrix-free transported chart, float64
response geometry and TF32 disabled. All 16/16 seeds passed the frozen 12/16
joint gate; recomputed medians give direction cosine 0.999990, active residual
ratio 1437.844, transverse gain contrast 341,763 and finite/JVP ratio 0.999998.

These ResNet audits close the finite-chart support for the paper's moving
geometry sections. They do not establish arbitrary-path, complete-kernel,
continuum or global variational claims.

## Main result II: low-response learning freedom

The geometric distinction has a functional consequence at recorded optimizer
nodes. Under four prospectively frozen finite response budgets, candidate
scales are selected on calibration data and scored once on disjoint held-out
data. The instantaneous response-kernel arm is compared with the recorded
optimizer update, the source kernel, a half-path time-shifted kernel and a
deterministic signed permutation.

Low first-order response cost is partly induced by the kernel construction and
is not itself the result. The nontrivial evidence is retained held-out learning
utility, calibration-to-held-out generalization and superiority over the
prospectively declared controls under the same finite budgets and alpha grid.

### Reduced ResNet v4.6 - supported

- 7/8 prospective seeds passed the complete frozen ordering.
- Current positive: 51/56 noninitial nodes.
- Current > source: 51/56.
- Current > half-path shifted: 49/56.
- Current > signed permutation: 52/56.
- Pooled median AUC: current 0.000790, source 0.000681, half-path shifted
  0.000728.

### GPT-2 native-LoRA-B v1-r1 - confirmed within the declared protocol

- 8/8 prospective seeds passed.
- All 24/24 noninitial nodes were positive and beat all declared controls.
- Pooled median AUC: current 0.11858; source 0.03341; half-path shifted
  0.03641; recorded AdamW 0.00851; signed permutation 0.00853.

### GPT-2 native-LoRA-B R2 strict-control upgrade - confirmed

- 8/8 prospective GPU seeds completed.
- 6/8 seeds passed the complete frozen control family.
- Current-minus-random-current-kernel contrast was positive in 7/8 seeds and
  20/24 noninitial nodes.
- Bootstrap 95% confidence interval for the mean seed contrast: [0.0296,
  0.1283]; exact two-sided sign-flip \(p=0.03125\).

R2 adds a calibration-selected best-of-16 equal-norm random direction inside
the same instantaneous response kernel. It therefore distinguishes
task-specific direction selection from generic current-kernel membership.

Recorded AdamW supplies real training nodes and proposals, but the successful
instantaneous-kernel arm is a counterfactual projection. Ordinary optimizer
navigation is not established.

### Secondary evidence: cached intrinsic Picard CIFAR v0.2.6

Picard v0.2.6 is a separate frozen-feature ResNet-18/CIFAR-10 timing audit in
a 20-dimensional float64-certified intrinsic response kernel. It compares a
cached intrinsic Picard update with AdamW using frozen v0.2.5 learning rates,
a frozen strict validation-loss target and five deterministic timing repeats
per seed.

All five new evaluation seeds passed every frozen timing, endpoint,
accuracy-noninferiority and response-leakage gate. Median time-to-equal-loss
speedup was 24.46%, median fixed-budget speedup was 16.70%, median
steps-to-target reduction was 9.09%, median endpoint loss delta was -0.000099
and certified response leakage was 8.23e-16. The status is
`PICARD_V0_2_6_REPEATED_TIMING_DUAL_10PCT_SPEEDUP_SUPPORTED`.

This earlier CIFAR result is secondary cross-model support for the declared
cached intrinsic Picard protocol only. It is not end-to-end fine-tuning, not
the GPT-2 LoRA r4/r5 result above, not a universal optimizer comparison and
not a proof of global Picard flow.

### Predictive and causal boundary audits

The R2 strict-control confirmation is accompanied by add-only predictive and
causal boundary audits. These audits ask whether the local low-response
opportunity can be promoted to stronger optimizer-future or causal-capacity
claims.

- PC1: static instantaneous capacity predicting subsequent AdamW gain is not
  supported.
- PC2: optimizer-access source is archived, but the authoritative result
  artifact is still required before any public repository-backed claim.
- PC3: capacity history and kernel drift forming a predictive state is not
  supported.
- PC4: response-matched high capacity causally raising subsequent gain is not
  supported.

These negative boundary results do not negate R2. They preserve the narrower
interpretation: the confirmed object is a counterfactual finite-budget
held-out frontier at recorded nodes, not a law for the future selected by
ordinary AdamW.

## Limited cross-modal confirmation: GPTW correction-cost scaling

This section concerns current-versus-source-fixed correction-cost scaling and
is independent of the low-response Pareto audit above.

GPTW v1.1.0 tests the moving-response-fibre mechanism in a deliberately
restricted language-model domain: the rank-2 LoRA-B subspace of the final two
GPT-2 transformer blocks. The same-seed precision repair passed in 7/8 seeds,
the prospective adaptive-value confirmation passed in 8/8 new seeds, and the
current-versus-source-fixed moving-fibre audit passed in 6/6 instances and
18/18 interior nodes. A final prospective test with disjoint frozen
natural-English development and response prompts passed in 8/8 new seeds and
32/32 interior nodes.

The v2/v3 current-versus-source-fixed audits are calibrated in
`docs/GPTW_SCALING_INTERPRETATION.md` and the v1.1.1
protocol-completeness addendum. The 2-versus-1 exponent split is analytically
forced by the arm definitions and retained as a numerical-correctness check,
not an independent discovery. The substantive v2 empirical quantities are the
finest-radius cost ratio 8.60--25.70, principal angle 0.0161--0.0713 rad,
active-J residual amplification about 1.4e4--1.4e5, maximum path-response error
6.7e-16 below the 2e-4 gate, and 6/6 seeds with 18/18 nodes passing. For v3
natural text, the corresponding substantive evidence is finest-radius ratio
8.87--35.56, active-J residual amplification 1.41e7--1.12e8, maximum
path-response error about 1.08e-15, and 8/8 seeds with 32/32 nodes passing.

Keep these counts separate: GPTW-v3 correction-cost audit uses 8/8 seeds and
32/32 interior nodes, while the GPT-2 Pareto audit uses 8/8 seeds and 24/24
noninitial nodes. The formal initial audit and its near-zero-denominator
finite-difference failure are retained in the snapshot.

This is limited cross-modal confirmation of the restricted mechanism, not a
full-model GPT-2 result, a semantic invariance theorem, an arbitrary-LoRA
claim, ordinary optimizer navigation, or a global variational theorem.

### Zero-upload external reproduction

GPTW-v3 has a public one-cell Colab entry point under `external_tests/gptw_v3/`.
It clones the immutable GPTW source tag, verifies the pinned commit and
snapshot checksums, runs the frozen natural-text V3 protocol, validates the
8-seed/32-node output, and downloads the result and environment manifest. No
author ZIP upload is required. This is explicitly a same-cohort external
reproduction, not a new-seed independent confirmation.

## Restricted variational audit and strict failure boundary

The restricted F16 audit tests a possible path-ordering functional; it is not
the second main result and does not supply the foundation of the moving-fibre
mechanism.

v3.2c froze six causal online algorithms and four Fisher step radii
\(h\in\{0.08,0.04,0.02,0.01\}\), giving 384 attempted paths across 16 new
seeds. Fourteen seeds were fully comparable, exactly meeting the frozen
eligibility requirement. In all 14 comparable seeds:

- natural gradient minimized the declared restricted action in 14/14
  comparable seeds at all four radii;
- natural gradient was not the least expensive path to realize;
- median correction-cost ratio was 2.25, range 1.60-4.19;
- action minimization and realization-cost minimization therefore cannot be
  identified;
- no global variational law is established.

True natural gradient beat wrong-Fisher natural gradient at every radius; all
six algorithms passed positive cost-scaling, fit-quality and fine-radius
action-convergence gates; current response-row-space rotation was detected; and
true natural gradient won under the wrong metric in 0/14 seeds. The median
natural-gradient realizability-cost exponent was 0.716 (range 0.573--0.871;
median \(R^2=0.993\)).

Two seeds, 73730 and 73732, were excluded because the coarse-radius natural arm
failed the frozen wrong-metric left-versus-trapezoid quadrature gate. Their
other core path diagnostics remained numerically regular. They are retained in
the raw and extracted results.

### Moving-Fibre F16 action used in v3.2c

For a retracted discrete path \(\theta_0,\ldots,\theta_K\), the reported action
is

\[
S_{\rm MF16}^{\rm trap}=\frac12\sum_{k=0}^{K-1}
\left(\frac{d\ell_k^-}{H_k}+\frac{d\ell_k^+}{H_{k+1}}\right),
\]

where \(\Delta\theta_k=\theta_{k+1}-\theta_k\),

\[
d\ell_k^-=\sqrt{(B_k^T\Delta\theta_k)^T G_k(B_k^T\Delta\theta_k)},\qquad
d\ell_k^+=\sqrt{(B_{k+1}^T\Delta\theta_k)^T G_{k+1}(B_{k+1}^T\Delta\theta_k)},
\]

and

\[
H_k=h_0+\frac{\sqrt{g_k^T G_k^{-1}g_k}}{s_{\rm true}},\qquad
g_k=B_k^T\nabla L_{\rm dev}(\theta_k).
\]

Here \(B_k\) is the transported eight-dimensional chart reprojected into
\(\ker DR(\theta_k)\), and \(G_k\) is the regularized output-Fisher matrix in
that chart. The left rule \(\sum_k d\ell_k^-/H_k\) is also computed; arm
admissibility requires its relative discrepancy from the trapezoidal rule to
be at most 0.08. This documents the frozen finite-path implementation, not a
continuum or arbitrary-path theorem.

## Confirmed evidence

| Evidence item | Frozen result | Formal status |
|---|---|---|
| GPT-2 LoRA Picard vs AdamW | r4 frozen five-new-seed confirmation passed 5/5 seeds; r5 ten-step audit reports 37.46% median time-to-equal-loss speedup, 5/5 positive seeds and 1.17% fixed-budget advantage as diagnostic only | Supported GPT-2 LoRA Picard confirmation plus resolution audit |
| Selected response-fibre directions carry adaptive value | ResNet-18/CIFAR-10 v4.0c-r1 passed all gates in 16/16 seeds at 4, 16 and 32 anchors | Confirmed fixed-chart functional premise |
| Moving-fibre finite realizability-cost scaling | CNN--MNIST v3.1e passed 15/15 comparable seeds; ResNet v4.1b passed 16/16 seeds with slope gap 0.5751 and smallest-radius cost ratio 2.997 | Confirmed finite-radius scaling |
| Transverse response amplification | ResNet v4.2d passed 16/16 seeds; median active residual ratio 1437.844 and transverse gain contrast 341,763 | Confirmed finite-chart transverse amplification |
| Low-response held-out learning frontier | Reduced ResNet v4.6 is supported with 7/8 seeds. GPT-2 native-LoRA-B R1 passed 8/8 seeds and 24/24 noninitial nodes; R2 strict-control upgrade passed 6/8 complete-control seeds with 7/8 positive same-kernel seed contrasts and 20/24 positive noninitial nodes | Supported in reduced ResNet; confirmed within the declared GPT-2 native-LoRA-B R1 and R2 protocols |
| GPTW current-versus-fixed correction-cost scaling | GPTW v2 is SUPPORTED in 6/6 seeds and 18/18 nodes; v3 natural text is SUPPORTED in 8/8 seeds and 32/32 nodes as not confined to codeword prompts | Supported restricted GPTW evidence |
| Pretrained GPT-2 dynamic response-kernel ordering | R13 passed 5/5 untouched seeds at both frozen global response budgets; current-kernel updates beat source-frozen and response-budgeted AdamW controls, while unconstrained AdamW is retained as an out-of-budget diagnostic | Confirmed within the declared pretrained GPT-2 LoRA response-budget protocol |
| Pythia-160M metric-constrained Onsager implementation | R17d passed the frozen aggregate gate with 4/5 untouched supporting seeds; median AdamW-minus-metric-Onsager loss +0.00135338 and source-frozen-minus-current loss +0.01599610 under one global response-budget ceiling | Confirmed model-specific variational implementation bridge |
| Restricted Moving-Fibre F16 ordering | Moving-Fibre F16 v3.2c passed with natural minimizing the restricted action at all four radii in 14/14 comparable seeds and winning under the wrong metric in 0/14 | Confirmed restricted multi-radius ordering |
| Action/realization-cost equivalence - Refuted | Natural gradient was not the least-cost path in 14/14 comparable seeds; smallest-radius correction-cost ratio median 2.25, range 1.60-4.19 | Refuted within the frozen six-algorithm family |

## Developmental and failed stages

Earlier v3.1a--e development and repair history remains included. In
particular, v3.1d is not erased: its inappropriate fixed-arm high-\(R^2\) veto
is documented, and v3.1e changed that statistic before observing a new cohort.

ResNet v4.4-r1 is staged only as a developmental frozen-code/protocol record
under `evidence/developmental/resnet18_cifar10_real_adam_path_v4_4_r1/`. It
tests whether ordinary AdamW updates align with current rather than
source-frozen response geometry. The A100 run record indicates that 8 seeds and
128 audited nodes completed before a post-outcome packaging error
(`NameError: name '__file__' is not defined`), but the authoritative recovered
results ZIP has not been imported. No positive or negative v4.4-r1 decision is
repository-backed here.

## Evidence map

```text
evidence/
  gpt2_lora_r2_strict/
    overlay-format R2 strict-control confirmation
  predictive_causal_boundaries/
    pc1/ static predictive-capacity negative audit
    pc2/ optimizer-access source only; result artifact required
    pc3/ dynamic/history predictive-state negative audit
    pc4/ response-matched causal-capacity negative audit
  picard_finetune_v0_2_6/
    cached intrinsic Picard CIFAR timing audit; secondary supported side branch
  gptw_response_fibre_v1_1_0/
    experiments/
    development/
    paper/
  gptw_response_fibre_v1_1_1/
    V2 protocol-completeness documentation only
  low_response_pareto_v1/
    resnet_v4_6/
    gpt2_lora_b_v1/
    gpt2_lora_b_r2_strict/
  pretrained_gpt2_dynamic_kernel_r12_r13/
    R12 development chain and R13 frozen pretrained GPT-2 confirmation
  fibre-qwen/
    Qwen v0.0.1-development protocols, fail-closed evidence and roadmap
  confirmed/
    pythia160m_metric_onsager_r17d/
      R17d five-untouched-seed metric-constrained Onsager confirmation
    picard_gpt2_lora_r4/
    f16_v16/
    moving_f16_v3_0b/
    moving_fibre_v3_1e/
    moving_fibre_f16_v3_2c/
    resnet18_cifar10_fibre_v4_0c_r1/
    resnet18_cifar10_dual_scaling_v4_1b/
    resnet18_cifar10_transverse_v4_2d/
  developmental/
    pythia160m_metric_onsager_r17_chain/
      R17a/R17b development and R17c same-seed budget calibration
    pythia160m_response_action_r16b/
      response-occupancy-only action negative boundary
    picard_gpt2_lora/
      r1/
      r2/
      r3/
    moving_fibre_v3_1a/
    moving_fibre_v3_1b/
    realisability_cost_v3_1c/
    realisability_cost_v3_1d/
    moving_fibre_f16_v3_2a/
    moving_fibre_f16_v3_2b/
    resnet18_cifar10_real_adam_path_v4_4_r1/
docs/
  GPTW_SCALING_INTERPRETATION.md
  CLAIM_BOUNDARIES.md
  EVIDENCE_LADDER.md
  robustness_v3_1e/
  supplementary/
audits/
  picard_gpt2_lora_r5_ten_step/
  formula_audit_v1/
  resnet_v4_1b_v4_2d/
provenance/
  PROVENANCE.md
  SHA256SUMS
external_tests/
  picard_gpt2_lora/
  gptw_v3/
  gpt2_lora_pareto/
paper/
  Moving_Response_Fibres_A_Geometric_View_of_Behaviour_and_Learning.pdf
  archive/Moving_Response_Fibres_v5_unified_hierarchy.pdf
```

Every confirmed stage preserves executable source, its frozen protocol,
machine-readable results and/or the original result ZIP. Python cache files are
excluded.

## Claim boundary

The archive supports a state-dependent moving-response-fibre mechanism with
two experimentally distinct consequences. First, stale and instantaneous
response charts differ in finite realizability cost, and small chart drift can
be transversely amplified. Second, the instantaneous response kernel defines a
superior held-out low-response learning frontier within the declared reduced
ResNet and GPT-2 native-LoRA-B protocols.

GPTW-v3 supplies a separately frozen restricted correction-cost confirmation.
The F16 programme supplies a strict boundary: restricted action minimization
does not coincide with realization-cost minimization.

The archive does not establish:

- a complete high-dimensional response-kernel bundle;
- exact \(h\to0\) convergence or a horizontal-lift theorem;
- local or global minimality over arbitrary causal paths;
- a unique scalar combination of F16 action and realizability cost;
- unrestricted ResNet backbone adaptation, ResNet F16 ordering or an
  arbitrary-path/global variational result;
- full-model GPT-2/LLM confirmation, semantic invariance, or transfer beyond
  the declared LoRA-B subspace and frozen prompts;
- ordinary AdamW or SGD naturally follows a response fibre;
- predictive capacity, optimizer-access, dynamic-state or causal-capacity laws
  for ordinary optimizer futures;
- universal optimizer superiority, end-to-end Picard fine-tuning, or global
  Picard-flow convergence;
- a continuous Pareto optimum beyond the frozen finite budget and alpha grids;
- Qwen personalization, safety certification, trained-model release or
  moving-response-fibre optimizer evidence from the fail-closed
  `fibre-qwen/` development overlay;
- a universal intelligence, physical-time or K=1 law.

## Reproduce

Each stage's `code/` directory contains its launcher, frozen `protocol.json`,
main program and shared F16 engine where required. GPU execution is strongly
recommended for v3.1e and v3.2c. Raw result ZIPs are authoritative; extracted
JSON directories are supplied for inspection.

Low-response Pareto reproduction has three layers:

```bash
python verify_low_response_snapshot.py
python verify_picard_finetune_v0_2_6.py
python external_tests/gpt2_lora_pareto/COLAB_ONE_CLICK_GPT2_LORA_PARETO.py --source-root . --smoke
python external_tests/gpt2_lora_pareto/COLAB_ONE_CLICK_GPT2_LORA_PARETO.py --source-root . --full
```

The publication-ready external-confirmation record is under `docs/si_v4_0c/`.
The authoritative raw output is preserved under
`evidence/confirmed/resnet18_cifar10_fibre_v4_0c_r1/raw/raw_results.zip`.

The publication-ready v3.2c run record is under `docs/si_v3_2c/`. It contains
the 384 algorithm-radius path rows, all 16 seed-level eligibility/gate rows,
and the two excluded-arm diagnostics required by Supplementary Item 13.

## Citation and license

Cite the associated paper version and the repository commit or release that
contains this archive. The repository includes a `LICENSE` file documenting
the current reuse boundary for this evidence package.
