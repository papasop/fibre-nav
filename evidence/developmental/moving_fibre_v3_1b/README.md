# Moving-response-fibre v3.1b deep-path audit

v3.1b tests whether the v3.1a signal survives paths deep enough to accumulate
nontrivial response-fibre motion. It uses eight new seeds, restores the 20%
capability-loss target, and reduces the per-step Fisher radius from 0.12 to
0.04. Natural paths must contain at least eight steps and the median depth of
the six algorithms must reach ten steps in at least six seeds.

At every online update the audit recomputes the current response Jacobian,
transports an eight-dimensional kernel chart by projection and Procrustes
alignment, retracts the finite update to the frozen anchor response, and
recomputes the output-Fisher pullback and capacity dual norm.

New controls beyond v3.1a:

- pure moving-Fisher length is reported separately from weighted action;
- a natural-gradient replay keeps the source chart fixed while retaining the
  same online batches, response retraction, target and local metric updates;
- full response-Jacobian row-space change is measured through principal angles
  and normalized projector distance, independently of the selected 8D chart;
- a structural-difference gate requires the moving and fixed natural actions
  to differ by at least 2% in six of eight seeds (no direction is assumed).

This remains an eight-seed developmental depth audit, not a 16-seed formal
confirmation, continuum horizontal lift, arbitrary-path minimum, GPT-2 result,
or global variational law. An A100 is strongly recommended.
