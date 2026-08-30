---
title: "Moving Response Fibres in Neural Networks"
subtitle: "Supplementary Information: frozen ResNet evidence for Sections 5.3--6"
author: "Y. Y. N. Li"
date: "Repository release v1.5.0"
geometry: margin=0.82in
fontsize: 10pt
header-includes:
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{microtype}
---

# Scope and evidential status

This supplement closes the archival record for the two load-bearing ResNet
results in Sections 5.3 and 6 of the paper. It records the frozen parameter
domains, cohorts, data-access boundaries, numerical gates, failure history,
per-seed reconstruction fields, and exact repository paths. The authoritative
scientific outputs are the original result ZIPs; expanded JSON and CSV files
are inspection aids.

The two results have different parameter domains and must not be conflated.
Section 5.3 trains a terminal residual adapter and the classifier over a frozen
ImageNet-pretrained ResNet-18 backbone. Section 6 trains ResNet-18 `layer4` plus
the classifier. Neither result concerns the complete backbone, a complete
response-kernel bundle, arbitrary paths, or a global variational principle.

# SI-R1. Section 5.3: cross-model cost-scaling confirmation

## SI-R1.1 Frozen construction

Protocol identifier:
`CNER_RESNET18_CIFAR10_MOVING_FIBRE_DUAL_SCALING_CONFIRM_V4_1B`.
The prospective cohort consists of seeds 68726--68741. The ImageNet-pretrained
ResNet-18 backbone is frozen; the permitted trainable domain is the terminal
residual adapter plus the complete classifier. The response declaration uses
16 anchors and the transported chart has dimension eight.

For each seed the current-kernel, source-fixed and transport-shuffled arms are
run at radii 0.08, 0.04, 0.02 and 0.01, with at most 48 steps and a common 3%
development-loss-reduction endpoint. All twelve development paths are frozen
before the first confirmation-set access. Confirmation data cannot select an
arm, radius, direction, step, endpoint or geometric gate.

## SI-R1.2 Frozen gates

At least 12/16 seeds must be geometric candidates. The per-seed gates are:

| Gate | Frozen rule |
|---|---:|
| Maximum response-retraction error | at most 0.002 |
| Moving-arm exponent | at least 0.40 |
| Moving-minus-fixed exponent gap | at least 0.40 |
| Smallest-radius fixed/moving cost ratio | at least 2.0 |
| Censored moving/shuffled step ratio | at least 2.0 |
| Moving/shuffled progress ratio | at least 1.5 |
| Projected development/confirmation gradient cosine | at least 0.95 |
| Moving held-out confirmation change | non-worsening |

The first six gates determine the geometric candidate. The two held-out gates
are reported separately and cannot alter the prospectively frozen geometric
decision.

## SI-R1.3 Reconstructed result

All 16 seeds passed every geometric and held-out gate.

| Quantity | Reconstructed median |
|---|---:|
| Moving-minus-fixed slope separation | 0.575101900 |
| Smallest-radius fixed/moving cost ratio | 2.996962733 |
| Raw development/confirmation gradient cosine | 0.414313068 |
| Projected development/confirmation gradient cosine | 0.996302975 |

These values are computed from the sixteen `seed_*.json` files, not copied
from `report.json`. The complete per-seed table is
`tables/SI_RESNET_V41B_SEEDS_16.csv`.

## SI-R1.4 Interpretation boundary

The result confirms a finite-radius separation between current-kernel
transport and source-fixed replay in this ResNet construction. It does not
establish a universal exponent, an asymptotic limit, full-backbone transport,
Moving-Fibre F16 ordering, or variational optimality.

# SI-R2. Section 6: transverse response amplification

## SI-R2.1 Frozen construction

Protocol identifier:
`CNER_RESNET18_CIFAR10_FULL_LAYER4_TRANSVERSE_CONFIRM_V4_2D`.
The prospective cohort consists of seeds 76742--76757. The trainable parameter
domain is ResNet-18 `layer4` plus `fc`. The response chart has dimension eight
and uses eight anchors. All response-geometry and audit operations are float64
with TF32 disabled. Matrix-free JVP/VJP block conjugate gradients use ridge
$10^{-5}$, at most 48 CG iterations and tolerance 0.002.

The three path radii are 0.004, 0.002 and 0.001. Finite/JVP consistency is
audited at relative micro-scales 1, 0.25, 0.0625 and 0.015625. Paths use at
most 24 steps and a common 3% development-loss-reduction endpoint.

At a current path state, let $d_m$ be the normalized descent proposal in the
transported, reprojected chart and $d_f$ the proposal in the source-fixed
chart. With current response Jacobian $J$ and $\delta d=d_f-d_m$, the audit
separates

$$
\frac{\|Jd_f\|}{\|Jd_m\|}
\quad\text{from}\quad
\Gamma_\perp=
\frac{\|J\delta d\|/\|\delta d\|}{\|Jd_m\|/\|d_m\|}.
$$

The first is an active residual ratio; the second is a transverse gain
contrast. They are never treated as the same statistic.

## SI-R2.2 Frozen joint gate

At least 12/16 seeds must pass all eleven conditions:

1. source chart feasible;
2. moving charts numerically feasible;
3. all moving endpoints reached;
4. smallest-radius moving path has at least three steps;
5. median chart-residual ratio fixed/moving at least 50;
6. median direction cosine at least 0.999;
7. median active residual ratio at least 50;
8. median transverse gain contrast at least 1000;
9. median finest finite-difference/JVP ratio in [0.9, 1.1];
10. at least 75% of audited states improve toward linearity with radius;
11. maximum JVP additivity relative error at most $10^{-8}$.

## SI-R2.3 Reconstructed result

All 16 seeds passed the full joint gate.

| Quantity | Reconstructed median |
|---|---:|
| Moving/fixed direction cosine | 0.999989986 |
| Active residual ratio | 1437.843934 |
| Transverse response gain | 90.108233 |
| Transverse gain contrast | 341,763.1519 |
| Finest finite-difference/JVP ratio | 0.999998021 |
| Per-seed maximum JVP additivity error | $5.05\times10^{-14}$ |

The complete per-seed table is `tables/SI_RESNET_V42D_SEEDS_16.csv`. Each raw
seed record additionally retains every path, every audited active state, the
finite-radius ladder, retraction error, CG residual and endpoint status.

## SI-R2.4 Precision failure and repair

The first development attempt passed its scientific signal gates but failed
the preregistered JVP identity certification under float32 with TF32 matrix
multiplication enabled. It is not counted as positive evidence. The repair
changed the audit precision to float64 and disabled TF32, then ran a fresh
sixteen-seed confirmation cohort. This failure-to-repair history is part of
the method boundary: numerical certification is a condition of admissibility,
not a post-hoc cosmetic check.

## SI-R2.5 Interpretation boundary

The result shows that nearly collinear proposals can differ by more than three
orders of magnitude in active response residual because their small difference
falls in a high-gain transverse direction. Finite-difference/JVP agreement
rules out the reported effect being merely a first-order algebra mistake. It
does not establish a global singular-spectrum law, a complete kernel bundle,
or a cost or action optimum.

# SI-R3. Per-layer judgment table

| Evidence layer | Parameter domain | Frozen outcome | Permitted conclusion |
|---|---|---|---|
| ResNet v4.0c-r1 | Complete final classifier; backbone frozen | 16/16 at 4, 16 and 32 anchors | Selected response-fibre tangent has held-out adaptive value |
| ResNet v4.1b | Terminal residual adapter + classifier; backbone frozen | 16/16 geometric and held-out pass | Current transport and stale replay have separated finite cost scaling |
| ResNet v4.2d | Trainable layer4 + classifier | 16/16 joint pass | Minute chart drift can be transversely amplified |
| GPTW v3 | Rank-2 LoRA-B in final two GPT-2 blocks | 8/8 seeds; 32/32 nodes | Restricted cross-modal replication on frozen natural text |
| Moving-Fibre F16 v3.2c | Eight-dimensional CNN--MNIST chart | 14/14 eligible positive gate | Restricted six-algorithm action ordering |

The ladder is cumulative but not interchangeable. In particular, v4.2d does
not inherit the F16 ordering, and the GPT-2 audit does not inherit the ResNet
parameter domain.

# SI-R4. Reconstruction and file identity

From repository root, run:

```bash
python audits/resnet_v4_1b_v4_2d/recompute_resnet_sections.py
sha256sum -c provenance/SHA256SUMS
```

The reconstruction command writes two per-seed CSV files and
`SI_RESNET_SECTIONS_5_3_6_SUMMARY.json`. It checks that exactly 16 records are
present for each protocol and recomputes every displayed median from those
records.

Authoritative locations:

The v4.1b root is
`evidence/confirmed/resnet18_cifar10_dual_scaling_v4_1b/`; the v4.2d root is
`evidence/confirmed/resnet18_cifar10_transverse_v4_2d/`. Within each root,
`code/` is the frozen executable package, `raw/raw_results.zip` is the
authoritative output, and `results/` is the expanded output.

# SI-R5. Final exclusions

The released ResNet evidence does **not** establish: unrestricted backbone
navigation; exact $h\to0$ convergence; a smooth global response-fibre bundle;
local or global minimality over arbitrary causal paths; ResNet F16 ordering;
equivalence of action and realizability cost; or a universal learning law.
Those remain open questions rather than implied consequences of Sections
5.3--6.
