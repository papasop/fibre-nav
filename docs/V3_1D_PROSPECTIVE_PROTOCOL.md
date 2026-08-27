# v3.1d Prospective Confirmation Protocol

This is the next proposed confirmation step. It is not evidence yet.

## Core claim

Freeze the moving response fibre as:

```text
B(theta) = ker DR(theta)
```

For matched endpoint changes, fixed-source-chart replay should have a stable
realisation-cost separation from online moving-fibre paths:

```text
C_fixed(h) / C_moving(h) > 1
```

The scaling law to test is:

```text
C_moving(h) ~ h^alpha_mov
C_fixed(h) ~ h^alpha_fix
```

v3.1c observed a median fixed/moving cost ratio near 10 and separated scaling
indices. v3.1d should test whether that candidate survives prospective
confirmation.

## Frozen design

- 16 new prospective seeds.
- Path depth at least covering the v3.1b deep-path interval.
- Step sizes: `h = {0.08, 0.04, 0.02, 0.01}`.
- Moving-kernel retraction and fixed-source replay use the same normal drive.
- Matched endpoint task improvement, for example 20%.
- Each step uses only current and past information.

## Pointwise certification

At each step certify:

```text
dtheta_parallel in ker DR(theta)
```

Record:

- endpoint residual;
- tangent-space residual;
- Fisher cost;
- realised retraction cost;
- scaling exponent per seed;
- wrong-kernel negative control;
- shuffled-kernel negative control.

## Passing gates

All gates must be written into the executable `protocol.json` before running.

- At least 12/16 seeds show the preregistered fixed-over-moving cost separation.
- At least 12/16 seeds show scaling-exponent separation.
- Wrong-kernel and shuffled-kernel negative controls fail the claimed moving
  fibre separation.
- No post-hoc threshold fitting.

If v3.1d passes, the repository can move from finite restricted algorithm
ordering toward a realisability-cost law for online response-fibre motion.
