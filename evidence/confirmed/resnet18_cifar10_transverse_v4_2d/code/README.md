# ResNet-18/CIFAR-10 transverse-amplification confirmation v4.2d

This is the frozen large-seed confirmation authorized by the v4.2c-r1
precision audit. It changes only the seed cohort and aggregate confirmation
threshold. The float64 geometry, disabled TF32, model domain, chart, paths,
radii, micro-radius ladder and every per-seed gate are unchanged.

Sixteen new seeds (76742--76757) are evaluated. A seed is a complete candidate
only if it passes numerical feasibility, endpoints, multistep identification,
chart separation, near-collinearity, active-response separation, transverse-
gain contrast, JVP additivity, finite-radius validity and radius convergence.
At least 12/16 complete candidates are required.

The principal quantities are

```text
delta_d = d_fixed - d_moving
A_perp = ||J delta_d|| / ||delta_d||
gain_contrast = A_perp / ||J d_moving||.
```

Frozen gates include chart and active-response ratios at least 50, direction
cosine at least 0.999, gain contrast at least 1000, maximum JVP additivity
error at most 1e-8, finest finite/JVP ratio in [0.9,1.1], and at least 75% of
states improving toward linearity over `h,h/4,h/16,h/64`.

Passing confirms this finite-chart mechanism in the stated trainable ResNet-18
`layer4 + fc` domain. It does not establish a complete Jacobian singular-
spectrum theorem, cost advantage, global variational principle, LLM transfer
or universal law. Use an A100; allow approximately 4--10 hours.
