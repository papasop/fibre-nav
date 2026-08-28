# Supplementary Item 13: Moving-Fibre F16 v3.2c run record

## Frozen scope

v3.2c is a prospective sixteen-seed confirmation of restricted Moving-Fibre
F16 ordering in a frozen CNN--MNIST construction. Six causal online algorithms
were evaluated at four Fisher step radii, producing 384 attempted paths:

```text
seeds: 73726--73741
radii: 0.08, 0.04, 0.02, 0.01
algorithms:
  adam
  normalized_sgd
  normalized_momentum
  sign_gradient
  natural_gradient
  wrong_fisher_natural_gradient
common endpoint: 20% capability-loss reduction
```

Every proposal was generated in an online transported eight-dimensional chart
projected into the current response kernel. No future minibatch, future
Jacobian, final result, fitted lambda or post-hoc scalarization was used.

## Discrete action definition

Let \(\Delta\theta_k=\theta_{k+1}-\theta_k\) be the actual retracted step,
\(B_k\) the current transported chart, \(G_k\) the regularized output-Fisher
matrix in that chart, and \(g_k=B_k^T\nabla L_{\rm dev}(\theta_k)\). Define

\[
H_k=h_0+\frac{\sqrt{g_k^T G_k^{-1}g_k}}{s_{\rm true}},
\]

\[
d\ell_k^-=\sqrt{(B_k^T\Delta\theta_k)^TG_k(B_k^T\Delta\theta_k)},\quad
d\ell_k^+=\sqrt{(B_{k+1}^T\Delta\theta_k)^TG_{k+1}(B_{k+1}^T\Delta\theta_k)}.
\]

The reported quantity is

\[
S_{\rm MF16}=S_{\rm MF16}^{\rm trap}
=\frac12\sum_k\left(\frac{d\ell_k^-}{H_k}+\frac{d\ell_k^+}{H_{k+1}}\right).
\]

The code also computes \(S_{\rm MF16}^{\rm left}=\sum_kd\ell_k^-/H_k\).
The frozen quadrature gate requires the relative left-versus-trapezoid change
to be at most 0.08 for both the true and wrong metrics. This is a finite,
moving-chart discretization, not an asserted continuum functional identity.

## Frozen confirmation gates

### Seed eligibility

- source accuracy at least 0.90;
- source-chart kernel residual at most \(10^{-7}\);
- frozen output-Fisher metric certification passed;
- every one of the 24 algorithm-radius arms admissible;
- at least 14/16 seeds fully comparable.

### Arm admissibility

- common capability endpoint reached;
- response-retraction relative error at most 0.002;
- retraction-to-step ratio at most 0.20;
- moving-kernel residual at most \(10^{-5}\);
- effective metric rank at least 8;
- maximum step principal angle at most 0.80 rad;
- both true- and wrong-metric left-versus-trapezoid action changes at most
  0.08.

### Confirmation outcomes

At least 12 seeds were required for each positive condition:

- natural action minimum at all four radii;
- natural action minimum at the smallest radius;
- natural beats wrong-Fisher natural at all four radii;
- all six algorithms have retraction-cost exponent at least 0.50;
- all six exponent fits have \(R^2\ge0.80\);
- all six actions change by at most 0.10 between radii 0.02 and 0.01;
- current response-row-space rotation detected.

Natural gradient was allowed to win under the wrong metric in at most 2/16
seeds. Smallest-radius natural/minimum realizability-cost ratios were frozen as
descriptive outputs and did not enter the primary confirmation gate.

## Result

```text
MOVING_FIBRE_F16_V32C_PROSPECTIVE_CONFIRMATION_SUPPORTED
```

Fourteen of sixteen seeds were fully comparable, exactly meeting the frozen
eligibility requirement. Every positive gate passed in all 14 comparable
seeds. True natural gradient had minimum Moving-Fibre F16 action at every
radius in 14/14, beat wrong-Fisher natural at every radius in 14/14, and won
under the wrong metric in 0/14.

The natural-gradient realizability-cost exponent had median 0.716 and range
0.573--0.871; median fit quality was \(R^2=0.993\). Across radii and comparable
seeds, the median natural action advantage over the best competing algorithm
was 22.8%.

The descriptive cost branch separated from the action result. At the smallest
radius, natural gradient's retraction cost was at least 1.5 times the
algorithmic minimum in 14/14 comparable seeds. The ratio had median 2.25 and
range 1.60--4.19. Restricted action minimality therefore did not imply
realizability-cost minimality.

## Exclusions

Seeds 73730 and 73732 were excluded because `natural_gradient_r0` failed the
frozen wrong-metric left-versus-trapezoid quadrature gate at radius 0.08:

| Seed | Observed | Frozen maximum | True-metric change | Kernel residual | Response error | Retraction/step | Rank |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 73730 | 0.106763 | 0.08 | 0.025084 | 1.786e-8 | 2.865e-7 | 0.014662 | 8 |
| 73732 | 0.103576 | 0.08 | 0.016278 | 1.800e-8 | 3.739e-7 | 0.011307 | 8 |

Both reached the capability endpoint and retained regular source, kernel,
metric and response-retraction diagnostics. The predeclared quadrature veto
was nevertheless applied without relaxation.

## Machine-readable tables

- `SI_V32C_PATHS_384.csv`: one row per seed, algorithm and radius, including
  actions, Fisher lengths, retraction costs and all arm admissibility metrics;
- `SI_V32C_SEED_GATES_16.csv`: eligibility, winners, seed-level frozen gates,
  exponents and smallest-radius cost ratios;
- `SI_V32C_EXCLUSIONS_2.csv`: exact failed statistic and regular diagnostics
  for the two excluded coarse-radius arms.

The authoritative complete nested record remains
`evidence/confirmed/moving_fibre_f16_v3_2c/raw/raw_results.zip` and its extracted
`result.json` plus 16 seed JSON files.

## Claim boundary

This run confirms only a finite-model, finite-radius ordering over six causal
algorithms in transported eight-dimensional charts. It does not establish a
complete response-kernel bundle, exact continuum convergence, arbitrary-path
or global minimality, LLM transfer, or a universal learning law.
Equivalently, the confirmed statement is an ordering over \(\mathcal A_6\)
only; no statement is made about an argmin over arbitrary causal paths.
