# Moving-F16 v3.0b: 16-seed confirmation

This package tests whether the restricted F16 ordering survives when both the
output-Fisher metric and the capacity dual norm are recomputed pointwise along
each executable path.

## What changed after v3.0a-r1

The scientific action and controls are unchanged. The only methodological
repair is the source response-kernel chart:

- compact SVD estimates the response-Jacobian row space;
- the chart is projected into its numerical null space;
- three project--QR iterations enforce orthogonality and kernel membership;
- the internal residual must be at most `1e-7`;
- the pre-existing external kernel gate remains `1e-5`;
- a seed is rejected if the numerical null dimension is below 8.

This prevents a marginal chart-construction residual from deciding the result.

## Frozen confirmation

- 16 new seeds: 65726--65741;
- six online executable algorithm paths;
- moving output-Fisher and moving capacity dual norm;
- adaptive path integration up to four subdivisions;
- 24 independent KL perturbations at three path locations per algorithm;
- at least 14/16 seeds must be fully comparable;
- moving natural gradient must win at least 12/16;
- true-moving natural must beat wrong-Fisher natural at least 12/16;
- natural may win under the wrong-moving metric at most 4/16.

The result is a restricted algorithm-family test. It is not a proof over all
paths, a moving full-space quotient projector, or a universal learning law.

## Run

Upload the package to Colab and run
`COLAB_LAUNCHER_MOVING_F16_V3_0B.py`, or execute:

```bash
python cner_cnn_mnist_moving_f16_confirm_v3_0b.py \
  --protocol protocol.json
```

An A100-class GPU is recommended. This 16-seed moving-metric confirmation is
substantially heavier than v3.0a.
