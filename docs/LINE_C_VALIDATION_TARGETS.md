# Line C validation targets

LC1–LC6 identify capabilities; C35–C37 identify experimental runs.

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
