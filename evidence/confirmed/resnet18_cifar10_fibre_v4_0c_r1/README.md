# ResNet-18/CIFAR-10 external precision confirmation v4.0c-r1

This prospective 16-seed protocol repairs only the numerical certification
failure of v4.0b. It projects the development-loss gradient into the response
kernel of the complete 5,130-parameter linear classifier on top of a frozen
ImageNet-pretrained ResNet-18 representation.

The primary gate uses 16 anchors. Four- and 32-anchor results are frozen
sensitivity gates, preventing a positive result from depending only on one
especially wide response kernel. The learned tangent must beat sign reversal,
the best of 32 matched random kernel tangents, and a shuffled-target tangent by
predeclared effect sizes unchanged from v4.0b. The unconstrained ambient gradient is never a
competition arm.

The Jacobian, SVD projection, finite direction step, centered-response
difference, and confirmation loss are evaluated in float64. Kernel membership
is certified by both `||Jv||/(||J||_2 ||v||_2) <= 1e-12` and finite centered-
logit leakage no greater than `1e-9`. New seeds 64726--64741 are used; the
formal v4.0b failure is not reclassified.

Implementation revision r1 changes no scientific seed, gate, threshold, data
split, direction or comparison. It only creates matched random controls with
`dtype=w.dtype`, fixing the float32/float64 matrix-vector mismatch that stopped
v4.0c-r0 before any seed result was produced. The r0 runtime failure is not
reclassified as a scientific result.

This package does not update the ResNet backbone and does not test moving
kernels, realizability-cost scaling, Moving-Fibre F16 action, an LLM, or global
variation. Recommended runtime: A100, approximately 25-45 minutes depending on
CIFAR download and feature caching.
