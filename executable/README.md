# Restricted CNER-F Prospective Confirmation v16

This prospective 16-seed confirmation (18726--18741) freezes the v13--v15 categorical output-Fisher construction, certification gates, capability target, action definition, natural-gradient construction, and wrong-Fisher control before observing any v16 outcome.

Metric certification has three independent parts. First, 24 unseen chart perturbations spanning radii 0.005--0.02 test whether one-half the Fisher quadratic form predicts the true output-distribution KL divergence. Second, a positive rescaling of all eight conv2 channels with inverse compensation in the final classifier preserves logits; the pushed-forward Fisher chart metric must remain invariant. Third, the hidden-representation pullback is required to change under the same gauge transformation, serving as a negative control against arbitrary hidden coordinates.

Six executable paths are compared at the same 20% capability-loss-reduction target: Adam, normalized SGD, normalized momentum, sign-gradient, true Fisher natural gradient, and a wrong-Fisher natural gradient. The wrong metric has the exact same eigenvalue spectrum but reverses the association between Fisher geometry and the eight chart coordinates. Every path receives the same frozen 240-step budget, is truncated by bisection at its first crossing of the common target, and is evaluated by the true output-Fisher action. A non-hitting path remains blocking and is never extrapolated or inserted into a matched-endpoint comparison.

Primary integration is frozen adaptive refinement. Every action first uses T4/T8; if their relative change exceeds 2%, it escalates successively to T8/T16 and T16/T32. Failure at T16/T32 is inadmissible. All six named output-Fisher paths determine primary eligibility. Identity and hidden-representation metrics remain visible diagnostics but cannot veto the Fisher claim.

Confirmation requires true Fisher natural gradient to be restricted action minimum in at least 12/16 seeds with nonpositive median gap, beat the wrong-Fisher natural path in at least 12/16 with exact one-sided p below .05 and negative median paired action, while wrong-Fisher natural minima remain at most 8/16. Fisher metric recertification remains independent of all optimizer outcomes.

The Fisher metric is regularized only for action evaluation; KL and gauge tests use the raw metric. In chart coordinates, the declared capacity is

`H_F(z) = h0 + sqrt(dA_z^T G_F^{-1} dA_z) / sqrt(dA_0^T G_F^{-1} dA_0)`,

where `A` is the prospectively frozen development-probe cross-entropy objective, `h0=0.1`, and `G_F` is trace-normalized after flooring eigenvalues at `0.001` times the largest raw output-Fisher eigenvalue. The action is the refined midpoint quadrature of `sqrt(dz^T G_F dz)/H_F(z)`. This explicit formula is part of the protocol, not inferred from optimizer outcomes.

This remains a restricted frozen 8-dimensional TinyCNN/MNIST cohort. A pass confirms only the finite CNER-F ordering declared here; it is not CNER-S, a global path-space theorem, ordinary-training minimization, or a universal law.

Run `COLAB_LAUNCHER.py` in a Colab GPU notebook cell and upload this ZIP. The doubled executable budget and possible T32 refinement make v16 slower than v15.
