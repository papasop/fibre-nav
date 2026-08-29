# Moving Response Fibres in Neural Networks

## Confirmed Scaling of Realizability Cost

Repository-ready frozen evidence archive v1.4.0 for the associated paper by
Y.Y.N. Li. This archive separates prospectively confirmed results from
developmental mechanism studies and preserves failed or excluded cases rather
than relabelling them after inspection.

## Cross-modal confirmation: GPT-2 + native LoRA

GPTW v1.1.0 tests the moving-response-fibre mechanism in a deliberately
restricted language-model domain: the rank-2 LoRA-B subspace of the final two
GPT-2 transformer blocks. The same-seed precision repair passed in 7/8 seeds,
the prospective adaptive-value confirmation passed in 8/8 new seeds, and the
current-versus-source-fixed moving-fibre audit passed in 6/6 instances and
18/18 interior nodes. A final prospective test with disjoint frozen
natural-English development and response prompts passed in 8/8 new seeds and
32/32 interior nodes.

This is cross-modal confirmation of the restricted mechanism, not a full-model
GPT-2 result, a semantic invariance theorem, an arbitrary-LoRA claim, or a
global variational theorem. The formal initial audit and its
near-zero-denominator finite-difference failure are retained in the snapshot.

## External confirmation: selected response-fibre tangent value

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
leakage was 2.98e-13. This confirms selected tangent value in a frozen external
representation; it does not update the ResNet backbone or externally confirm
moving-kernel transport, cost scaling or Moving-Fibre F16 ordering.

## Main result I: moving-fibre realizability scaling

Functional freedom relative to a prospectively declared response is represented
locally by

\[
V_\theta=\ker DR(\theta).
\]

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
scaling in the frozen CNN--MNIST construction; it is not a continuum fibre-
bundle theorem.

## Main result II: Moving-Fibre F16 prospective confirmation

v3.2c froze six causal online algorithms and four Fisher step radii
\(h\in\{0.08,0.04,0.02,0.01\}\), giving 384 attempted paths across 16 new
seeds. Fourteen seeds were fully comparable, exactly meeting the frozen
eligibility requirement. In all 14 comparable seeds:

- true natural gradient had the minimum Moving-Fibre F16 action at every radius;
- true natural gradient beat wrong-Fisher natural gradient at every radius;
- all six algorithms passed positive cost-scaling, fit-quality and fine-radius
  action-convergence gates;
- current response-row-space rotation was detected;
- true natural gradient won under the wrong metric in 0/14 seeds.

The median natural-gradient realizability-cost exponent was 0.716
(range 0.573--0.871; median \(R^2=0.993\)). Its median action advantage over
the best competing algorithm across radii was 22.8%.

The result also preserves a nontrivial conflict: at the smallest radius,
natural gradient had persistently higher retraction cost in 14/14 comparable
seeds. Its cost relative to the least-cost algorithm had median 2.25 and range
1.60--4.19. Thus minimum restricted action is not the same as minimum
realizability cost.

Two seeds, 73730 and 73732, were excluded because the coarse-radius natural arm
failed the frozen wrong-metric left-versus-trapezoid quadrature gate. Their
other core path diagnostics remained numerically regular. They are retained in
the raw and extracted results.

### Moving-Fibre F16 action used in v3.2c

For a retracted discrete path \(\theta_0,\ldots,\theta_K\), the reported action is

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

| Stage | Frozen result | Formal status |
|---|---|---|
| GPTW v1.1.0 | Rank-2 LoRA-B adaptive value passed 8/8 new seeds; current-vs-fixed passed 6/6; disjoint natural-text confirmation passed 8/8 seeds and 32/32 nodes | Confirmed restricted cross-modal mechanism |
| ResNet-18/CIFAR-10 v4.0c-r1 | Selected final-classifier response-fibre tangent passed all gates in 16/16 seeds at 4, 16 and 32 anchors | Confirmed external functional premise |
| F16 v16 | True output-Fisher natural flow minimized the restricted six-path CNER-F action in 15/16 seeds; Adam and wrong-Fisher natural won 0/16 | Confirmed restricted ordering |
| Moving-F16 v3.0b | With pointwise moving Fisher and capacity evaluation, 14/16 seeds were comparable and natural minimized moving action in 13/14 | Confirmed restricted ordering |
| Moving-Fibre v3.1e | Current-fibre and stale-fibre replay had sharply separated realizability-cost scaling; 15/15 comparable seeds passed | Confirmed restricted scaling |
| Moving-Fibre F16 v3.2c | Fourteen seeds were comparable; natural minimized the restricted moving-fibre action at all four radii in 14/14 and won under the wrong metric in 0/14 | Confirmed restricted multi-radius ordering |

## Developmental bridge to v3.2c

| Stage | Role | Outcome |
|---|---|---|
| v3.2a | Four-seed same-domain action/cost Pareto preflight | Candidate supported; natural action won 4/4 but cost won 0/4 |
| v3.2b | Four-seed, four-radius discretization audit | Frozen gate passed in 3/4; authorized independent confirmation |
| v3.2c | Sixteen-seed prospective confirmation | Formal gate passed; 14/14 comparable seeds satisfied every positive gate |

Earlier v3.1a--e development and repair history remains included. In
particular, v3.1d is not erased: its inappropriate fixed-arm high-\(R^2\) veto
is documented, and v3.1e changed that statistic before observing a new cohort.

## Repository layout

```text
evidence/
  gptw_response_fibre_v1_1_0/
    experiments/
    development/
    paper/
  confirmed/
    f16_v16/
    moving_f16_v3_0b/
    moving_fibre_v3_1e/
    moving_fibre_f16_v3_2c/
    resnet18_cifar10_fibre_v4_0c_r1/
  developmental/
    moving_fibre_v3_1a/
    moving_fibre_v3_1b/
    realisability_cost_v3_1c/
    realisability_cost_v3_1d/
    moving_fibre_f16_v3_2a/
    moving_fibre_f16_v3_2b/
docs/
  CLAIM_BOUNDARIES.md
  EVIDENCE_LADDER.md
  robustness_v3_1e/
audits/
  formula_audit_v1/
provenance/
  PROVENANCE.md
  SHA256SUMS
```

Every confirmed stage preserves executable source, its frozen protocol,
machine-readable results and/or the original result ZIP. Python cache files are
excluded.

The publication-ready v3.2c run record is under `docs/si_v3_2c/`. It contains
the 384 algorithm-radius path rows, all 16 seed-level eligibility/gate rows,
and the two excluded-arm diagnostics required by Supplementary Item 13.

## Claim boundary

The archive supports state-dependent response fibres, confirmed finite-radius
realizability-cost scaling, and restricted six-algorithm Moving-Fibre F16
ordering in frozen finite CNN--MNIST constructions. It also confirms selected
response-fibre tangent value in the complete final classifier of a frozen
ResNet-18 representation and provides prospective cross-modal confirmation in
the rank-2 LoRA-B subspace of the final two GPT-2 blocks. It does not establish:

- a complete high-dimensional response-kernel bundle;
- exact \(h\to0\) convergence or a horizontal-lift theorem;
- local or global minimality over arbitrary causal paths;
- a unique scalar combination of F16 action and realizability cost;
- ResNet backbone adaptation or cross-architecture confirmation of moving-fibre
  transport, realizability scaling or v3.2c;
- full-model GPT-2/LLM confirmation, semantic invariance, or transfer beyond
  the declared LoRA-B subspace and frozen prompts;
- a universal intelligence, physical-time or K=1 law.

## Reproduction

Each stage's `code/` directory contains its launcher, frozen `protocol.json`,
main program and shared F16 engine where required. GPU execution is strongly
recommended for v3.1e and v3.2c. Raw result ZIPs are authoritative; extracted
JSON directories are supplied for inspection.

The publication-ready external-confirmation record is under `docs/si_v4_0c/`.
The authoritative raw output is preserved under
`evidence/confirmed/resnet18_cifar10_fibre_v4_0c_r1/raw/raw_results.zip`.

## Citation and license

Cite the associated paper version and the repository commit or release that
contains this archive. The repository includes a `LICENSE` file documenting
the current reuse boundary for this evidence package.
