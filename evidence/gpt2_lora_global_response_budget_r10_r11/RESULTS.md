# Frozen result summary

## R10 development

R10 reused the R7–R9 development seeds intentionally. The current-kernel arm
was positive at all five seeds and all four global response budgets, nominating
the protocol for untouched-seed confirmation. This stage is not confirmatory.

## R11 untouched-seed confirmation

| Budget | Median source-current validation loss | Current-positive seeds | Frozen gate |
| ---: | ---: | ---: | :---: |
| `5e-5` | `0.003762603` | `5/5` | PASS |
| `1e-4` | `0.003669620` | `5/5` | PASS |
| `2e-4` | `0.003578544` | `5/5` | PASS |
| `5e-4` | `0.003186703` | `5/5` | PASS |

Confirmation seeds: `27211`, `27217`, `27229`, `27241`, `27253`.

All declared gates passed:

- five untouched confirmation seeds;
- finite losses in every run;
- every global response budget respected;
- float64 projector eligibility;
- projector idempotence at most `1e-10`;
- constant response rank; and
- support at no fewer than three of four budgets (observed: four of four).

Scientific status:
`R11_CURRENT_KERNEL_GLOBAL_BUDGET_PARETO_CONFIRMED`.

The five seeds—not the 20 correlated seed-by-budget contrasts—are the primary
independent experimental units.
