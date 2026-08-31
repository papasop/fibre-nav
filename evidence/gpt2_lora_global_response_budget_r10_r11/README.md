# GPT-2 LoRA global response-budget experiment: R10/R11

R10 developed a comparison between a response kernel recomputed at the current
parameters and a kernel frozen at the source. Both arms used the identity
metric and were constrained at every accepted step by

```math
\lVert R(\theta_t)-R(\theta_{\mathrm{warm}})\rVert \le B.
```

R11 froze the full R10 configuration and changed only to five untouched seeds.
The confirmation gate required at least three of four budgets to have a
positive paired median and at least four of five current-positive seeds, while
all numerical and budget gates passed. R11 supported all four budgets with
five of five positive seeds at each budget.

See the bundle-level `README.md` for reproduction and `CLAIM_BOUNDARY.md` for
the precise scientific scope.
