# Line C: training qualification, frozen replay and pending confirmation

Evidence snapshot: 28 October 2026. **LC2 remains open.** C196–C198 concern
the same 84 exposed training states; C199 is a delivered prospective protocol
without an execution result in this import.

## What changed

The supplied C196 log reports complete native acceptance at the predetermined
final head: **83/84 → 84/84 in both rounds**, one rescued state and no destroyed
states. The rescued state is `xor_11__c141_confirmation_p08_n01_d01`.
Historical `confirmation` IDs remain exposed training data.

C197 reports a completed pure-NumPy audit of C196's eight updates, 840 saved
FP64 points, 336 native endpoint records and all 84 trajectories. It retains
`C196_FIXED_FINAL_TRAINING_ALL84_PASS_NO_CONFIRMATION`; it does not perform
new model measurements.

C198 reports a new physical CPU GPT-2 run with the final writer frozen. Both
the original archived endpoint path and the path using newly acquired prewrite
features, recomputed routing and newly generated proposals pass **84/84 in each
of two rounds**. Preparation checks also pass84/84 in both rounds. It records
168 fresh feature captures, 672 ordered records and 1513 model forwards, with
zero backward calls, fitting, training or repair. This supplies reported
same-programme model-level reproducibility, not external-laboratory replication
or new-state/excluded-input acceptance.

| Run | Decision in supplied log | Evidence available in this commit |
| --- | --- | --- |
| C196 | `C196_FIXED_FINAL_TRAINING_ALL84_PASS_NO_CONFIRMATION` | Original delivered source/BUNDLE, supplied execution log and exact extracted summary |
| C197 | `C197_READONLY_CONTINUATION_TRAIN_AUDIT_COMPLETE_PARENT_DECISION_RETAINED` | Original delivered source/BUNDLE, supplied execution log and exact extracted summary |
| C198 | `C198_FROZEN_TRAINING_REPLAY_ALL84_PASS_NO_GENERALISATION` | Original delivered source/BUNDLE, supplied execution log and exact extracted summary |
| C199 | Execution pending | Frozen protocol, original delivered source/BUNDLE and software-validation record only |

## Numerical scope

The fixed response budget is0.002. C196/C198's final native maximum is
0.0019989013671875, with minimum slack approximately1.10e-6. The C196 FP64
response-only maximum is0.001994054531834877; **FP64 full acceptance was not
evaluated**. Precision agreement is not an accuracy bound.

C198's newly acquired features are not identical to the archived features:
maximum capture/full-input difference is approximately0.02288818, maximum raw
coordinate difference0.000358213, maximum routing-weight difference0.000106577,
and maximum final-point coordinate difference0.000358414. Actual final gates
still pass. These descriptive differences do not establish a hardware cause,
universal robustness or a bound on systematic forward error.

The response map, KL anchor distribution, readouts, update domain and precision
remain protocol-specific. Passing declared aggregate response gates does not
protect arbitrary outputs. Neither training acceptance nor these two repeated
rounds establishes generalisation; two rounds are not168 independent states.

## Next test: C199

C199 freezes the same final C196 head and complete proposal pipeline. It
registers48 new directions around12 known preparation centers,16 per trained
class, with the historical norms0.125 and0.375 and two directions per norm.
Generation uses fixed seeds and the original shared-row orthogonal complement;
this is not a certified local tangent space or finite-write guarantee.

All coordinates are sealed before model use. Bound-archive ID/seed/hash checks
do not prove programme-wide non-access. A complete84-state known preflight
precedes new-state measurements. If it fails, the conditional protocol stops
without claiming confirmation. Otherwise all48 new preparations and proposals
are evaluated twice; failures remain in the48-state denominator. No resampling,
recalibration, training, alternative head or repair is allowed. The complete
path requests624 records and1429 CPU model forwards.

**C199 has no scientific execution result here.** Even an all48 pass would
support local new-direction confirmation in represented classes, not excluded
inputs or LC2 closure. Excluded-input isolation and historical-access disclosure
remain separate requirements under the [existing validation contract](LINE_C_VALIDATION_TARGETS.md).

## Provenance and reproduction

The [evidence directory](../evidence/gpt2_lc2_c196_c199/) contains supplied logs,
log-derived summaries, protocol snapshots, exact hashes and a read-only checker.
The [CPU reproduction directory](../external_tests/gpt2_lc2_c196_c199/) contains
the four original launcher/BUNDLE pairs and one unchanged expanded source tree.

**No original scientific result ZIP was reopened or published during this
import.** Their hashes and owner-side Drive paths identify dependencies, not
public downloadable attachments. C197's reported archive audit does not upgrade
this import to an independent audit of those absent bytes. The log/source
checker does not replay GPT-2 or inspect the absent actual final-head tensors.

Original C35–C37 scope and pending independent confirmation, earlier failed
decisions, B5 integration and LC3–LC6 statuses remain unchanged. Offline C196
Jacobians, LPs and training costs are not correction-free online execution.
