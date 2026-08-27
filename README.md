# Moving Response Fibres in Neural Networks

## Confirmed Scaling of Realizability Cost

Repository-ready frozen evidence archive v1.2 for the associated paper by
Y.Y.N. Li. This archive separates prospectively confirmed results from
developmental mechanism studies and preserves failed or excluded cases rather
than relabelling them after inspection.

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

## Confirmed evidence

| Stage | Frozen result | Formal status |
|---|---|---|
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
  confirmed/
    f16_v16/
    moving_f16_v3_0b/
    moving_fibre_v3_1e/
    moving_fibre_f16_v3_2c/
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
provenance/
  PROVENANCE.md
  SHA256SUMS
```

Every confirmed stage preserves executable source, its frozen protocol,
machine-readable results and/or the original result ZIP. Python cache files are
excluded.

## Claim boundary

The archive supports state-dependent response fibres, confirmed finite-radius
realizability-cost scaling, and restricted six-algorithm Moving-Fibre F16
ordering in frozen finite CNN--MNIST constructions. It does not establish:

- a complete high-dimensional response-kernel bundle;
- exact \(h\to0\) convergence or a horizontal-lift theorem;
- local or global minimality over arbitrary causal paths;
- a unique scalar combination of F16 action and realizability cost;
- GPT-2, LLM or cross-architecture confirmation of v3.2c;
- a universal intelligence, physical-time or K=1 law.

## Reproduction

Each stage's `code/` directory contains its launcher, frozen `protocol.json`,
main program and shared F16 engine where required. GPU execution is strongly
recommended for v3.1e and v3.2c. Raw result ZIPs are authoritative; extracted
JSON directories are supplied for inspection.

## Citation and license

Cite the associated paper version and the repository commit or release that
contains this archive. No license is selected by this evidence package; the
repository owner must add an explicit `LICENSE` before inviting reuse or
redistribution.
