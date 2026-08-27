# Moving-fibre realizability-cost audit v3.1c

v3.1c tests the hidden constraint-maintenance work discovered in v3.1b. It
does not fit a post-hoc coefficient or declare a new composite action.

For each of eight new seeds, four natural-gradient paths reach the same 20%
capability-loss target:

- moving response kernel, Fisher step radius 0.04;
- moving response kernel, Fisher step radius 0.02;
- fixed source chart plus response retraction, radius 0.04;
- fixed source chart plus response retraction, radius 0.02.

Every arm uses the same online batches, target, metric probes, response anchors,
Newton retraction, and endpoint bisection. The audit records:

- cumulative Euclidean retraction norm (diagnostic);
- cumulative retraction Fisher length from local output KL;
- cumulative pre-retraction response error;
- tangent residual and retraction-to-tangent ratio;
- coarse/fine action convergence;
- full response-row-space rotation.

The frozen mechanism prediction is asymptotic. A truly current tangent has a
per-step retraction of order h^2, so cumulative retraction to a matched endpoint
should decrease approximately as h. A stale fixed chart generically has an
order-h normal component, so its cumulative correction need not vanish as h is
halved. The audit therefore compares the coarse/fine scaling exponents without
introducing an arbitrary lambda-weighted penalty.

This is an eight-seed developmental mechanism audit, not a new action law,
16-seed confirmation, exact continuum theorem, arbitrary-path result, or LLM
transfer. An A100 is strongly recommended.
