# New random-A chart seeds — full protocol failed

Protocol and executable code were frozen before inference at GitHub commit
`8294a3e9add1a9ec92f66cb4c6c35fa7e2df89a0`. Same model revision, prompts,
readout and budgets as L1 development; only random-A chart seeds changed.
The initial B is zero, so these are new chart initializations, not independent
base models, tasks or datasets. CPU runtime: 108.97 seconds.

| Seed | WRITE | OVERWRITE | Outcome |
| --- | --- | --- | --- |
| 84031 | margin 1.1213; KL .004448 | margin 1.7862; KL .004824 | full program pass |
| 84047 | margin .2183; KL .009959 | not run | WRITE rejected and rolled back |
| 84061 | margin .8106; KL .009995 | not run | WRITE rejected and rolled back |

Overall **1/3** programs passed, so the predeclared all-three gate failed.
Two failures approached the KL limit .01 without meeting margin >= 1; finite
backtracking found no further admissible improving step among its eight scales.
This does not prove that no feasible solution exists in those charts.

For the two accepted operations in seed 84031, all six random controls failed.
Best random target margin was -.81346 for WRITE and -1.08804 for OVERWRITE;
true margins were 1.12127 and 1.78623. Both no-move controls failed. Maximum
endpoint-norm mismatch was 3.58e-7, below the 1e-5 gate. The aggregate control
gates in report.json are false because they also require six qualified true
operations; they must not be read as observed norm-matching failures.

Controls are local to each accepted true operation, from its exact start.
They normalize random directions in the starting near-kernel to the true
committed endpoint displacement norm. They have no retraction/backtracking and
are not optimizer trajectories. Failed true operations have no qualified random
comparisons. Do not report 0/18 controls: only 6 controls were evaluated.
No statistical significance, universal superiority or L2 claim is made.

Original development and checkpoint reload evidence remain valid in their
restricted scope; this new result blocks a three-seed stability claim.
Next: investigate gradient directions under the active KL constraint; any change
must receive a new development protocol and retain these failures unchanged.
