# Result ledger

## Frozen R13 confirmation

R13 changed only to five untouched seeds: `33211`, `33217`, `33229`, `33241`
and `33253`. The two global response budgets, 300 steps, warm start, chart,
LoRA rank, adapted layers, learning rates and backtracking rule were frozen
from R12b.

| Budget | Median current loss | Median source loss | Median budgeted-AdamW loss |
| ---: | ---: | ---: | ---: |
| `2e-5` | `4.372774482` | `4.374151707` | `4.374183973` |
| `5e-5` | `4.372774482` | `4.374109348` | `4.374171813` |

Current had lower loss than both constrained controls in all five seeds at
both budgets. Every numerical, determinism, response-rank and global-budget
gate passed.

## Mechanism and timing diagnostics

At `2e-5`, median zero-step fractions were `0.000` for current, `0.550` for
source-frozen and `0.873` for response-budgeted AdamW. Median complete arm
times were respectively `30.5 s`, `125.9 s` and `167.2 s`. At `5e-5`, the
corresponding values were `0.000`, `0.593`, `0.883` and `30.5 s`, `131.0 s`,
`166.7 s`.

These timings include current-kernel Jacobian/SVD work. They are implementation
times under the declared global response balls, not a universal per-step
complexity result.

## Unconstrained AdamW scope diagnostic

Unconstrained AdamW reached a lower median validation loss (`4.369313081`) but
its median response drift was about `1.78e-2`: approximately 890 times the
`2e-5` budget and 356 times the `5e-5` budget. It is excluded from the matched-
budget confirmation gate and prevents an unconditional AdamW-superiority claim.
