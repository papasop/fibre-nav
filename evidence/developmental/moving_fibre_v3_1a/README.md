# Moving-response-fibre v3.1a quick preflight

This package tests the structural step beyond Moving-F16. The source chart is
not reused as the allowed direction space. At every online update the audit:

1. recomputes the current anchor-response Jacobian;
2. projects and Procrustes-aligns an eight-dimensional basis of its kernel;
3. constructs the algorithmic update inside that current kernel;
4. applies a causal Newton response retraction after the finite step;
5. recomputes the local output-Fisher pullback and capacity dual norm; and
6. integrates the true and wrong moving-fibre actions.

Subspace motion is measured with principal angles, not raw basis differences,
so sign and orthogonal gauge choices do not count as fibre rotation.

This is a four-seed developmental preflight. It is intentionally not a formal
Moving-F16 replacement or a global variational claim. Passing it authorizes a
16-seed confirmation with tighter integration and retraction tolerances.

Run the Colab launcher in the package. An A100 is strongly recommended: exact
response Jacobians are recomputed repeatedly along all six online paths.
