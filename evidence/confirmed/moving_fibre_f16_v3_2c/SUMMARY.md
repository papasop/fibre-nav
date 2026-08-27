# Moving-Fibre F16 v3.2c result summary

Formal status:

```text
MOVING_FIBRE_F16_V32C_PROSPECTIVE_CONFIRMATION_SUPPORTED
```

- attempted seeds: 16;
- fully comparable: 14 (frozen minimum: 14);
- natural action minimum at all four radii: 14/14 comparable;
- natural beats wrong-Fisher natural at all radii: 14/14;
- all-algorithm positive retraction scaling: 14/14;
- all-algorithm fit quality and fine-radius convergence: 14/14;
- natural wins under wrong metric: 0/14;
- natural retraction exponent median: 0.716, range 0.573--0.871;
- natural scaling fit median R-squared: 0.993;
- median natural action advantage over the best competitor: 22.8%;
- smallest-radius natural cost at least 1.5 times the minimum: 14/14;
- smallest-radius natural/minimum cost ratio: median 2.25, range 1.60--4.19.

Seeds 73730 and 73732 were excluded because `natural_gradient_r0` failed the
frozen wrong-metric left-versus-trapezoid quadrature gate at the coarsest
radius. The original seed JSON files remain in the result archive.

This confirms a restricted finite-model ordering, not arbitrary-path or global
minimality.
