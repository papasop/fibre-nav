# Formula audit v1

This audit records assumptions needed when formulas are restated outside the
frozen Euclidean implementation.

1. For \(P^g=I-g^{-1}DR^T(DRg^{-1}DR^T)^\dagger DR\), the identities
   \((P^g)^2=P^g\), \(DRP^g=0\), and \((P^g)^Tg=gP^g\) hold up to numerical
   tolerance.
2. The covariant normalized descent field is
   \[
   \dot\theta=-\frac{P^g g^{-1}\nabla L}
   {\|P^g g^{-1}\nabla L\|_g}.
   \]
   Away from fibre-critical points it satisfies
   \(dL/dt=-\|P^g g^{-1}\nabla L\|_g<0\) and \(dR/dt=0\).
3. At \(P^g g^{-1}\nabla L=0\), the normalized field is undefined; no strict
   descent statement is made there.
4. The equivalence \(\|DR[v]\|_\Pi=0\iff v\in\ker DR\) requires
   \(\|\cdot\|_\Pi\) to be a positive-definite norm on response space. It can
   fail for a degenerate seminorm.
5. The paper's executable construction uses \(g=I\), for which the Euclidean
   shorthand is valid. These qualifications do not modify the frozen result.

`recompute_radius_loo.py` independently reconstructs the post-confirmation
v3.1e LOO table from the archived seed JSON files.
