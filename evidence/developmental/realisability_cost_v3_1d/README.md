# Moving-fibre realizability-cost prospective confirmation v3.1d

This package prospectively tests the v3.1c scaling candidate on 16 new seeds
(69726--69741). It compares two causal online constructions at the same frozen
20% capability-loss endpoint:

- a moving response-fibre chart recomputed from the current response Jacobian;
- a fixed source chart followed by the same online Newton response retraction.

Both constructions receive identical past-and-current minibatches, response
anchors, Fisher probes, endpoint rule, and budgets. No future batch, future
Jacobian, or confirm outcome is used to construct a direction.

The frozen Fisher step radii are 0.08, 0.04, 0.02, and 0.01. A four-point
log--log regression replaces the two-point exponent used in v3.1c. The main
gate requires at least 14 comparable seeds and at least 12 seeds passing the
depth, quadrature, realizability-ratio, row-space-rotation, fit-quality, and
moving-versus-fixed exponent-separation gates.

The fixed source chart is the preregistered stale-chart counterexample. This
package does not introduce a fitted penalty coefficient or a new action law.
It confirms only a finite-model realizability-cost scaling statement if the
frozen gate passes. It is not an exact continuum theorem, arbitrary-path
global variation, GPT-2 transfer, or universal intelligence law.

Run `COLAB_LAUNCHER_REALISABILITY_V3_1D.py` in one Colab cell and upload the
package ZIP. Eight arms per seed and the 0.01-radius paths make this materially
heavier than v3.1c; an A100 is strongly recommended.
