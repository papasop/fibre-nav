## Pythia-160M constrained Onsager confirmation (R17d)

R17 tests whether the instantaneous response kernel can enter a prospectively
declared incremental variational update rather than only a counterfactual
finite-budget direction audit.  The learning target is prompted GLUE/SST-2;
the independently declared response map comprises four AG News topic-margin
coordinates on frozen, disjoint inputs.

For Adam first- and second-moment statistics \(\hat m_k,\hat v_k\), define
\(M_k=\operatorname{diag}(\sqrt{\hat v_k}+\epsilon)\).  The frozen candidate is
the exact solution of

\[
\min_{DR(\theta_k)\delta=0}
\left[\hat m_k^T\delta+\frac{1}{2\eta_k}\delta^TM_k\delta\right],
\]

namely

\[
\delta_k^*=-\eta_k\left[M_k^{-1}-M_k^{-1}J_k^T
(J_kM_k^{-1}J_k^T)^{-1}J_kM_k^{-1}\right]\hat m_k,
\qquad J_k=DR(\theta_k).
\]

R17c reused its development seed intentionally and selected the frozen step
multiplier 1.15 under a predeclared response-budget-utilization rule.  R17d
then evaluated that configuration once on five untouched seeds against
current-kernel projected AdamW and source-frozen metric Onsager under the same
global response-budget ceiling.

- 4/5 untouched seeds passed the complete frozen seedwise gate.
- Median AdamW-minus-metric-Onsager validation loss was +0.00135338.
- Median source-frozen-minus-current validation loss was +0.01599610.
- Median metric-Onsager-minus-AdamW accuracy was -0.390625 percentage points,
  inside the frozen -0.5-point noninferiority margin.
- Every numerical, response-budget, rank, KKT and finite-value gate passed.

The authoritative status is
`R17D_METRIC_CONSTRAINED_ONSAGER_CONFIRMED`.  This is a model- and
protocol-specific confirmation within pretrained Pythia-160M, one
32-dimensional LoRA chart, one learning task, one independent response map and
one global response budget.  It is not a continuous-action theorem, a
universal optimizer ordering, a Principle-R theorem or a physical law.  One
seed strongly favored projected AdamW, and the comparison is under a shared
budget ceiling rather than equal realized budget consumption in every seed.

R16b is retained as the strict negative boundary: the response-occupancy-only
path functional A1 is not stationary along three frozen feasible fields.
R17a--R17c are retained as development, metric diagnosis and same-seed budget
calibration; they are not relabelled as confirmation.
