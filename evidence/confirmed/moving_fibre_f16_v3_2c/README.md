# Moving-Fibre F16 v3.2c: prospective confirmation

This package freezes the v3.2b design and tests it on sixteen new seeds, 73726--73741. No result from these seeds is used to alter the algorithms, radii, metric, thresholds, or endpoint.

## Design

- Six causal online algorithms.
- Four frozen step radii: 0.08, 0.04, 0.02, and 0.01.
- 24 paths per seed; 384 paths in total.
- An eight-dimensional chart is transported and projected into the current response kernel at every step.
- All paths target the same 20% capability-loss reduction.
- No fitted lambda or post-hoc scalar composite.

At least 14/16 seeds must be fully comparable. The primary confirmation requires at least 12/16 seeds to satisfy every positive gate: natural-gradient action minimum at all four radii, natural beating wrong-Fisher natural at all radii, positive and well-fit retraction-cost scaling for all six algorithms, converged fine-radius action, and detectable fibre rotation. Natural may win under the wrong metric in at most 2/16 seeds.

The natural retraction cost relative to the minimum algorithm at the smallest radius is reported descriptively as near-minimum (at most 1.25) or persistently higher (at least 1.50); this distinction does not enter the primary gate.

A pass confirms only restricted six-algorithm Moving-Fibre F16 scaling in the frozen CNN--MNIST model. It is not arbitrary-path/global minimality, a continuum theorem, GPT-2 transfer, or a universal intelligence law.

## Colab

Run `COLAB_LAUNCHER_MOVING_FIBRE_F16_V3_2C.py`. This evaluates 384 moving-fibre paths and is substantially heavier than v3.2b; an A100 is strongly recommended.
