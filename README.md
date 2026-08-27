# Neural Fibre Geometry: frozen evidence archive v1.1

This repository-ready archive separates prospectively confirmed results from
developmental mechanism studies for *The Geometry of Functional Freedom in
Neural Networks*.

## Confirmed evidence

| Stage | Frozen result | Status |
|---|---|---|
| F16 v16 | In a frozen six-path TinyCNN/MNIST cohort, true output-Fisher natural flow minimized the restricted CNER-F action in 15/16 seeds; wrong-Fisher natural and Adam won 0/16 | Confirmed restricted ordering |
| Moving-F16 v3.0b | With the output-Fisher pullback and capacity dual norm recomputed pointwise, 14/16 seeds were comparable and true natural minimized moving action in 13/14; it won under the wrong moving metric in 0/14 | Confirmed restricted ordering |
| Moving-Fibre v3.1e | Across four frozen Fisher step radii and 16 new seeds, 15/16 were comparable and all 15 passed the moving-versus-fixed realizability scaling gate | Confirmed restricted scaling |

## Version status

| Version | Positioning |
|---|---|
| v3.1a | Fast developmental candidate. |
| v3.1b | Deep-path mixed result. |
| v3.1c | 8-seed scaling candidate. |
| v3.1d | Entity predictions all passed, but the frozen fixed-arm \(R^2\) gate design was not applicable. |
| v3.1e | Independent new-seed prospective confirmation after correcting the statistical gate before the new run. |

The v3.1e medians over the 15 comparable seeds were

\[
\alpha_{\rm moving}=0.684,\qquad
\alpha_{\rm fixed}=0.00715,
\]

with smallest-radius fixed/moving retraction-Fisher cost ratio 21.30 and
tangent-residual ratio 21.47. The moving four-point log--log fit had median
\(R^2=0.995\). The fixed four-point relative cost span had median 3.05%.

## Evidence ladder

| Stage | Role | Formal outcome |
|---|---|---|
| v3.1a | Four-seed moving-response-fibre preflight | Developmental quick candidate supported |
| v3.1b | Eight-seed deeper path audit | Full candidate gate not supported; action-separation gate failed |
| v3.1c | Eight-seed two-radius realizability scaling | Developmental scaling candidate supported |
| v3.1d | Sixteen-seed four-radius first confirmation | Formal gate failed solely because fixed-arm high-\(R^2\) was inappropriate for a predicted constant control; substantive predictions passed 16/16 |
| v3.1e | Independent new-seed zero-slope-aware confirmation | Confirmed: 15/15 comparable seeds passed every substantive gate |

v3.1d is preserved rather than erased. It records why v3.1e replaced the
ill-conditioned fixed-arm \(R^2\) veto with prospectively frozen tests of
absolute exponent and relative span on entirely new seeds.

## Layout

```text
evidence/
  confirmed/
    f16_v16/
    moving_f16_v3_0b/
    moving_fibre_v3_1e/
  developmental/
    moving_fibre_v3_1a/
    moving_fibre_v3_1b/
    realisability_cost_v3_1c/
    realisability_cost_v3_1d/
docs/
  CLAIM_BOUNDARIES.md
  EVIDENCE_LADDER.md
provenance/
  PROVENANCE.md
  SHA256SUMS
```

Each stage preserves executable source, its frozen protocol, extracted
machine-readable results, and/or the original result ZIP. Python cache files
are intentionally excluded.

## Scope

The confirmed results concern frozen finite TinyCNN/MNIST constructions. They
do not establish an exact continuum limit, global variational theorem,
universal learning law, GPT-2 transfer, physical-time law, or K=1
chronogeometrodynamics.

GPT-2 is intentionally kept outside the main evidence trunk. Its current status
is boundary evidence for a Fisher-distance phenomenon, not confirmed
capacity-weighted CNER or local variational minimality.

Author: Y.Y.N. Li. Cite the associated paper version and the repository commit
or release containing this archive.

## License

No license is selected by this evidence package. The repository owner must add
an explicit `LICENSE` before inviting reuse or redistribution.
