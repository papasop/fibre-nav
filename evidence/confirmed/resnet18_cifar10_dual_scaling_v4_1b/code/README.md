# ResNet-18/CIFAR-10 Moving-Fibre v4.1b

Prospective 16-new-seed confirmation of the dual-scaling separation developed
in v4.1a-r2. The ImageNet ResNet-18 backbone is frozen after feature extraction.
The audited 13,842-parameter domain is a trainable terminal residual adapter
plus the complete classifier.

For each seed, current-kernel moving transport, a source-fixed chart, and a
transport-shuffled current-kernel control are evaluated at radii 0.08, 0.04,
0.02, and 0.01. Every direction and stopping event uses the dev split only.
All 12 paths are generated and frozen before the first access to confirm data.

The predeclared geometric gates per seed are:

- moving and fixed reach the same 3% dev-loss endpoint at all four radii;
- response retraction error is at most 2e-3;
- moving cost exponent is at least 0.40;
- moving minus fixed exponent is at least 0.40;
- fixed/moving cost ratio at radius 0.01 is at least 2;
- moving is faster than shuffled at every radius;
- median censored shuffled/moving step ratio is at least 2;
- median moving/shuffled dev-progress ratio is at least 1.5.

Confirmation requires at least 12 of 16 seeds to pass every geometric gate.
The r2-development-only fixed-flat gate is not used: r2 prospectively rejected
that stronger model-specific hypothesis on all four development seeds.

Held-out generalization is reported separately. Its descriptive gates are a
dev/confirm fibre-projected source-gradient cosine of at least 0.95 and no
confirm-loss worsening for any moving path. Neither can alter the geometric
confirmation decision.

This is not full `layer4 + fc` transport, F16 action ordering, arbitrary-path
global variation, or a universal learning law. Recommended runtime: Colab A100.
Run the launcher as one notebook cell and upload this ZIP when prompted.
