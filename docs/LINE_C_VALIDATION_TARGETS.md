# Line C validation targets

LC1–LC6 identify six capabilities requiring separate acceptance, not a consecutive ladder
or a renumbering of historical L1–L5 or B4/B5. C numbers identify experiments.
Programmable parametric memory is the theme; LC2 is the near-term focus.
See the [unified evidence matrix](PARAMETRIC_MEMORY_EVIDENCE_MATRIX.md) for sources,
verification tiers, costs and missing evidence.

| Target | Status and next evidence |
| --- | --- |
| LC1: correction-free frozen writing | Local development acceptance reported by both composite arms in C35–C37. Independent archive audit and confirmation pending. |
| LC2: excluded-input generalisation | Open. Rebuild a writer with a prospectively fixed input-class exclusion from all fitting and selection. |
| LC3: capacity and cost scaling | Open. Measure capacity, interference and full costs under matched gates; include offline preparation, failures and verification. |
| LC4: cross-model and broader domains | Open. Re-declare coordinates, access and gates for each model/domain. |
| LC5: continual learning | Open. Test acquisition and retention across sequences with independent task evaluation. |
| LC6: update internalisation | Open. Attribute proposal, geometry, commitment and optimisation to model or runtime; verify any claimed internal mechanism. |

## LC2 boundary before the next experiment

The deployed C35–C37 core and calibration have seen all four XOR input classes.
Removing examples now does not create a valid excluded-input experiment. Existing
developed classes are inspected data, even if new local seeds are generated.

Fix a train-three/exclude-one split before training each fold. Reuse the frozen
GPT-2 backbone and declared chart/access rules if desired, but rebuild every
class-conditioned learned writer, normalisation, data-dependent geometry prior,
calibration and selection rule from training-only information. Any allowed shared
prior must be specified before access to held-out outcomes and documented.
The updater must support conditioning on an excluded request without a branch
trained or calibrated on that request. State preparation/evaluation geometry
must not enter the writer or its selection.

Freeze checkpoints and selection rules before evaluation. Report all requested
states, invalid preparations and failures, with raw, fixed-geometry and learned
arms separately. Do not refit, resample or switch branches based on held-out
outcomes. Report the four folds separately; same-backbone folds are not four
independent model replications. Start with a separately labelled feasibility
pilot if necessary, then prospectively freeze a confirmation protocol.

These targets are separable: LC1 does not establish LC2–LC6. A successful LC2
experiment also does not establish native parameter updates.

## Subsequent local records

[Locally inspected C38–C41 summaries](LINE_C_LOCAL_WRITING.md#research-frontier-and-subsequent-local-records)
do not change these targets: C38 fails its single excluded-input fold;
C39 is a training-only diagnostic; C40 and completed C41 fail their seen-class
residual-module development tests. LC2 remains open. Public archive import and
independent model reproduction remain pending.

## Current LC2 records and integration boundary

[C52/C54/C55 compact records](../evidence/gpt2_lc2_c52_c54_c55/) retain the historical
36-state seen-class development outcomes. C54 remains a development failure at
32/36; its four failures and unchanged gates are listed. C55 has a completed
read-only audit result, beyond implementation/self-test evidence, but no new model
execution, candidate or excluded-class evaluation. Its parent failure is unchanged.
Full original local ZIP member hashes were checked; full public archives and
independent replay remain pending. LC2 is open.

[Integrating the B5 controller with a frozen writer](B5_FROZEN_WRITER_INTEGRATION.md)
is a separate proposed acceptance test. C36/C37 do not establish controller
integration. This round specifies real reads, required output flip, unselected
retention, all endpoint gates and causal controls without running the experiment.

## Later development: training, seen-class confirmation and excluded inputs

[C141–C155](LINE_C_C141_C155.md) adds selected evidence without changing the six
separate targets. C141 retains its historical new-local, **seen-class** confirmation
role: C140 passes 42/48 in each of two rounds, with six response failures. C142
is a read-only audit preserving that failure, not independent model replay.
Those exposed states are no longer eligible for future unseen evaluation.

C148 and later use all 84 states as exposed **training** data even when IDs contain
`confirmation`. C148 global/routed arms both reach 79/84; C150 diagonal/cyclic
arms reach 79/84 and 80/84. Neither meets all-state training qualification; routing
and coupling comparisons also differ in parameter count (36/108 and 108/216).
These outcomes do not establish an excluded-input pass or routing superiority.

C154/C155 remain **log-level diagnostics**: six source BUNDLE identities do not
upgrade two missing result archives. C155 reports 756 zero-difference FP32
forward pairs, with only two FP64 focus states; this is not all-84 qualification,
FP64 ground truth or proof of AD error. LC2 is open and LC3–LC6 remain open.
Original failures, thresholds and final checkpoints are preserved; neither a
retry nor selecting a better iterate/norm can relabel the original proposal.
