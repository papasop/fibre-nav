# C52/C54 seen-class development and C55 read-only audit

These are byte-identical compact JSON copies from local result ZIPs, not a new
experiment. [Provenance](PROVENANCE.json) records exact filenames, sizes, ZIP and
member SHA-256 values. All 1,128 C52, 458 C54 and 89 C55 manifest members were
hashed during import; C55's parent ZIP identities match the local C52/C54 files.
No archived source or model was executed. Full ZIPs, source snapshots and tensor
payloads remain in the original local archives; this compact extraction cannot
resume or independently reproduce the experiments. Public full-result attachments
are pending; no download URL is asserted.

| Protocol | Original decision and result | Source |
| --- | --- | --- |
| C52 response-sensitive distillation | `C52_DEVELOPMENT_FAILURE`; baseline and sensitive arms both 32/36 | [Summary](c52/SUMMARY.json), [all endpoint rows](c52/EVALUATION_ROWS.json), [protocol](c52/PROTOCOL.json) |
| C54 cumulative response distillation | `C54_DEVELOPMENT_FAILURE`; replayed C52 reference and cumulative arm both 32/36; no rescued or destroyed passes | [Summary](c54/SUMMARY.json), [all endpoint rows](c54/EVALUATION_ROWS.json), [protocol](c54/PROTOCOL.json) |
| C55 CPU transfer audit | `C55_READONLY_TRANSFER_AUDIT_COMPLETE_C54_DECISION_RETAINED`; recomputed 32/36 in both arms | [Summary](c55/SUMMARY.json), [paired states](c55/PAIRED_STATES.json), [input immutability record](c55/READONLY_CHECK.json) |

All 36 states are historical seen-class development states. Six whole-norm folds
(train 30 / test 6) do not exclude an XOR input class. Trajectory samples and
intersecting training folds are not independent trials. C52/C54 have zero new
confirmation states and zero excluded-class states; LC2 remains open.

C55 has an actual completed result record, beyond a launcher or self-test. Its
recorded arithmetic audit reconstructs 72 paths and checks seven coupled equations;
this import verifies hashes and compact record consistency, **not** that full
arithmetic audit anew. It performs no independent model replay. Training proxy
improvement does not establish improvement on out-of-fold states or a causal
explanation. The terminal Jacobian is absent: 72 endpoint rows remain uncovered
by the pointwise linear diagnostic, though measured endpoint gates are recorded.

## Failures retained

Both C54 arms fail the same four states. The response budget stays `0.002`;
all 36 endpoint gates and all 17 sampled response points are required. No failure
is dropped and no threshold is changed.

| State | C52 reference endpoint response | C54 cumulative endpoint response | First sampled breach (both) |
| --- | ---: | ---: | ---: |
| `xor_00__c41_training_n04_d00` | 0.00336456298828125 | 0.00336456298828125 | 9 |
| `xor_00__c41_training_n05_d00` | 0.00414276123046875 | 0.004150390625 | 7 |
| `xor_01__c41_training_n05_d00` | 0.0026397705078125 | 0.0026092529296875 | 11 |
| `xor_11__c41_training_n04_d00` | 0.0022430419921875 | 0.00225067138671875 | 13 |

## Cost and verification scope

C52 reports 576 new offline Jacobians, 7 fits and 112 linear solves; geometry
uses 1,873 model-forward invocations and 648 metric evaluations. C54 reuses
geometry, adds 7 fits / 7 solves, and reports no solver retries, jitter, SVD or
hyperparameter search. Each reports 2,449 candidate-phase model-forward
invocations, 1,224 metric evaluations and 1,224 parameter assignments across the
two arms. These phase totals include verification, not just proposal latency.
Neither trains the backbone; zero backbone steps does not mean zero fitting cost.
C55 reports zero new model loads/forwards, fits, training steps, solves, inverses,
Jacobians or SVD calls. Its CPU I/O and arithmetic cost is not zero.

Parent preparation/teacher costs, complete retry histories, wall time, peak memory
and energy are not normalized here; missing costs are unknown, not zero. Account
for inherited development and failures before making efficiency comparisons.

From the repository root:

```bash
python3 evidence/gpt2_lc2_c52_c54_c55/verify_records.py
```

This checks copied hashes against original manifests, parent identities, recorded
numeric gates, counts and failure IDs. It does not read missing full archives,
verify raw model measurements, execute archived code or qualify LC2. Full-result
SHA-256 identities are in provenance for a future public release and independent
audit. [Unified matrix](../../docs/PARAMETRIC_MEMORY_EVIDENCE_MATRIX.md).
