# Programmable parametric memory: evidence matrix

The research theme is programmable parametric memory. B4/B5 provide operational
and causal foundations; LC1 supports local frozen writing; LC2 is the near-term
research focus. **LC1–LC6 have separate capability acceptance criteria, not a ladder whose
six levels have been achieved.** Historical L1–L5, B0–B5, C experiment IDs and V
versions retain their identities and are not renumbered into LC targets.

## Evidence tiers

- **Archived consistency verified:** original archive/member identities and saved
  records checked; does not mean independent model reproduction.
- **Log-supported:** execution logs and exact extracted summaries agree; missing
  full results/tensors prevent the stronger archive/weight claim.
- **Locally inspected:** named local sources reviewed, with public publication
  scope stated explicitly. A launcher or self-test alone is only implementation
  evidence, not a completed run.
- **Open/proposed:** no qualifying result in this matrix. Absence is not a pass.

## Capability matrix

| Capability definition | Supporting protocol / result | Applicable scope | Source | Verification status | Next missing evidence |
| --- | --- | --- | --- | --- | --- |
| B0–B3 foundations: accessible, selectively writable parameter contents | Historical L1–L5 protocols and shared-state audits; paper-scoped results | Separate GPT-2 LoRA charts and single-seed protocols, not a unified multi-seed result | [Paper source](../paper/A_Programmable_Address_Space_for_Neural_Memory.tex), [historical index](PROGRAMMABLE_PARAMETRIC_MEMORY.md#evidence-index-and-version-history) | Mixed historical archive availability; legacy checker has a missing console-log blocker | Missing historical originals and unified independent confirmation |
| B4: parameter contents causally participate in read–compute–write | Minimal causal XOR development pass; read-value substitution, stored-input substitution, reset and restore | One memory-chart development seed, input `01`; model reads, runtime XOR and numerical writeback | [Summary](../evidence/gpt2_b4_minimal_causal_a100_v1/recorded/SUMMARY.json), [cases](../evidence/gpt2_b4_minimal_causal_a100_v1/recorded/seeds/82931/cases.json) | Archived consistency verified; full B4 XOR V2 remains a distinct pending protocol | Independent memory seeds, broader causal coverage |
| B5: model selects operations | V2 two-model and V3 partitioned-adapter development; V4 2/4 failure followed by V5 4/4 combined coverage | Runtime routes adapters, executes XOR and commits numerical updates. 8/8 readiness is symbolic, not eight physical experiments | [V2](../evidence/gpt2_b5_trained_dispatch_a100_v2/), [V3](../evidence/gpt2_b5_single_backbone_a100_v3/), [V4](../evidence/gpt2_b5_xor_truth_table_a100_v4/), [V5 origins](../evidence/gpt2_b5_xor_continuation_a100_v5/recorded/CONTINUATION_PLAN.json) | Archived consistency verified; inherited and continued development, not independent B5 confirmation | Fresh evaluations, independent memory seeds and [frozen-writer integration](B5_FROZEN_WRITER_INTEGRATION.md) |
| LC1: frozen proposal without online correction | C35 both composites 16/16 new local states; C36 both 32/32 new local states; C37 both 32/32 shifted-centre states | Same model/chart, related cohorts; offline geometry, externally supplied output bit and runtime parameter commitment. Fixed geometry passes too; learned calibration superiority/necessity not established | [C35 summary](../evidence/gpt2_line_c_c35_c37/reported/c35_summary_from_log.json), [C36 summary](../evidence/gpt2_line_c_c35_c37/reported/c36_summary_from_log.json), [C37 summary](../evidence/gpt2_line_c_c35_c37/reported/c37_summary_from_log.json), [limits](LINE_C_LOCAL_WRITING.md) | Log-supported local development acceptance; C36 is completed as reported, not execution-pending | Full result ZIPs and actual frozen tensors, independent audit and prospective confirmation |
| LC2: transfer to requests excluded from fitting and selection | Open. C38 excluded-input failure; C39 diagnosis; C40/C41 seen-class failures. C52 and C54 each 32/36, development failures; C55 completed read-only audit retains C54 failure | C52/C54: historical seen classes, six whole-norm folds; zero excluded-class states. C55 adds no new candidate or model evaluation | [C38–C41 local provenance](LINE_C_LOCAL_WRITING.md#research-frontier-and-subsequent-local-records), [C52/C54/C55 compact records](../evidence/gpt2_lc2_c52_c54_c55/) | C38–C41 local-summary tier; C52/C54/C55 original ZIP member hashes and compact records checked, full public attachments pending; no independent model replay | Training-only fitting/selection, frozen excluded-request protocol and prospectively evaluated states across declared folds |
| LC3: capacity and cost scaling | Open; no accepted scaling result identified here | Capacity, interference and costs must share declared gates and comparable budgets | [LC3 target](LINE_C_VALIDATION_TARGETS.md) | No qualifying result in this matrix | Capacity sweep, matched baselines and complete cost accounting including failures |
| LC4: cross-model/domain transfer | Open for this memory/writer capability | Other geometry or Qwen experiments do not confirm this writer | [LC4 target](LINE_C_VALIDATION_TARGETS.md), [separate foundations](RESPONSE_FIBRE_FOUNDATIONS.md) | No qualifying cross-model writer result | Declared model/domain, chart/access rules and fresh evaluations |
| LC5: continual acquisition and retention | Open | Finite local retention or one output flip is not continual learning or zero forgetting | [LC5 target](LINE_C_VALIDATION_TARGETS.md) | No qualifying sequential result | Long sequences, interference and independent task retention measures |
| LC6: update internalisation | Open | Proposal, XOR, geometry, optimizer and commitment must be attributed separately | [LC6 target](LINE_C_VALIDATION_TARGETS.md), [integration responsibilities](B5_FROZEN_WRITER_INTEGRATION.md) | Current execution still uses external runtime | Evidence for a specified internal update mechanism and its causal controls |

## Selected later LC2 evidence

[Main entry and interpretation](LINE_C_C141_C155.md). Four original ZIPs and all
manifest members were checked **during packaging**; this import checks the
provided compact members, actual tensors and arithmetic. The full original ZIPs
are absent here. Six unchanged source BUNDLE identities are checked separately;
that source check does not upgrade C154/C155 result evidence. No stage below
closes LC2 or changes LC1/LC3–LC6 status.

| Stage / capability question | Result and applicable scope | Source | Verification tier | Next missing evidence |
| --- | --- | --- | --- | --- |
| C141 / local confirmation of frozen C140 | 42/48 in each of two seen-class rounds; six response failures; no excluded inputs evaluated. Historical role retained; now-exposed states cannot be reused as unseen | [Original summary](../evidence/gpt2_lc2_c141_c155/records/c141/SUMMARY.json) | Archive-backed compact records and actual frozen tensors; gate/count arithmetic checked | Full public parent chain and independent model replay; new, prospectively declared capability evaluation |
| C142 / audit of confirmation | Retains C141 failure; no new model forwards | [Original summary](../evidence/gpt2_lc2_c141_c155/records/c142/SUMMARY.json) | Archive-backed read-only audit records and C141 linkage | Independent model replay is not supplied by arithmetic audit |
| C148 / context routing | Global and routed both 79/84 in each round; exposed training; 36/108 parameters confound comparison; no routing advantage or certified atlas | [Original summary](../evidence/gpt2_lc2_c141_c155/records/c148/SUMMARY.json) | Archive-backed records, router/features/heads/coordinates; proposal reconstruction checked | Complete training qualification and matched comparisons before new capability evaluation |
| C150 / cyclic coupling | Diagonal 79/84, cyclic 80/84 in each round; four cyclic response failures; 108/216 parameters; training qualification fails | [Original summary](../evidence/gpt2_lc2_c141_c155/records/c150/SUMMARY.json) | Archive-backed compact tensors and endpoint arithmetic | Outstanding training failures, matched capacity comparison, new frozen evaluation |
| C154 / fixed-direction numerical diagnostic | Completed according to supplied log; finite scale behaviour does not prove AD error | [Log-derived summary](../evidence/gpt2_lc2_c141_c155/records/c154/SUMMARY_FROM_LOG.json) | Source plus execution log only; raw result not audited | Result supplement with original hashes, all records and source identity checked |
| C155 / forward-route precision diagnostic | Log reports 756 FP32 pairs with zero differences; FP64 covers two focus states, not all 84 or ground truth | [Log-derived summary](../evidence/gpt2_lc2_c141_c155/records/c155/SUMMARY_FROM_LOG.json) | Source plus execution log only; no independent backward verification | Verified result supplement; any broader claim requires its own test |

All later 84-state cohorts are training data regardless of `confirmation` in
IDs. Do not discard failures, relax thresholds, pick a best iterate after seeing
outcomes, or count a retry as original-proposal success. Full replay dependencies
remain external; [dependency identities](../evidence/gpt2_lc2_c141_c155/REPLAY_DEPENDENCIES.json)
are not downloadable result attachments. C154/C155 supplements must be added
separately and checked before changing evidence tier; current logs remain intact.

## Current training milestone: C196–C199

[Stage guide](LINE_C_C196_C199.md) · [Logs and provenance](../evidence/gpt2_lc2_c196_c199/) · [CPU entry points](../external_tests/gpt2_lc2_c196_c199/).
This import checks original delivered source/BUNDLE identities and exact supplied
execution-log summaries. **The three scientific result ZIPs were not reopened
or published here.** A report that C197 audited42 originals is not an import-time
verification of those originals. C198's reported real-model replay is likewise
separate from this repository's read-only log/source check.

| Stage | Reported outcome | Included evidence / remaining boundary |
| --- | --- | --- |
| C196 | 83/84 →84/84 in both native rounds; one rescue, no destruction; fixed final step8 | Log-supported training qualification. Offline derivatives and eight LP calls belong to preparation/training, not correction-free online execution. |
| C197 | Completed read-only C196 archive/arithmetic audit; parent decision retained | Log-supported audit completion; source and log present, audited result ZIP absent. No new model output measurements. |
| C198 | Both archived endpoints and regenerated fresh-feature proposals pass84/84 in both rounds;168 fresh captures,1513 model forwards, no training | Log-supported same-programme model replay. All84 are exposed; no new-state/excluded-input evaluation or external-laboratory replication. |
| C199 | Fixed48-state new-direction confirmation,12 known centers,16 per seen class, norms0.125/0.375, two rounds | Protocol/source only; execution pending. Conditional84-state preflight before any new-state measurement; maximum1429 model forwards. No result or LC2 pass. |

The final C196/C198 native response maximum is0.0019989013671875 against0.002,
leaving approximately1.10e-6 slack. New C198 features differ from archived
features, while actual final gates pass; this is not byte-identical proposal
reproduction or a numerical accuracy bound. FP64 full acceptance is not claimed.
All original failures and current exposure remain recorded. **LC2 is open.**

## B5 lineage and budget accounting

V4's original failure remains: 00/01 pass, 10 preparation fails, and 11 writeback
is rejected. V5 inherits 00/01 unchanged, resumes the real case-10 preparation
checkpoint at step 600 to 1200, and restarts case-11 writeback from its saved
preparation. It does not recover the lost rejected 480-step candidate. The V4
phase caps were preparation 600 / writeback 480; V5 expands them to preparation
1200 total / writeback 1080 per attempt. Both V5 new writes use 1080 steps.
The controller is not retrained and the writer source/read rules remain unchanged.

These are increased **compute budgets**, not relaxed acceptance gates: response
`0.002`, mean KL `0.005`, exact/development margins `0.05` remain unchanged.
Source: [V4 launcher](../evidence/gpt2_b5_xor_truth_table_a100_v4/source/launch_b5.py),
[V5 continuation plan](../evidence/gpt2_b5_xor_continuation_a100_v5/recorded/CONTINUATION_PLAN.json).
The controller seed 82933 and memory-chart seed 82931 are not backbone random
seeds. V3 controls are inherited, not rerun on every V5 input. B5 action selection
plus LC1 writer acceptance is **not** evidence of an integrated successful system.

## Cost ledger and gaps

| Evidence | Preparation / fitting | Execution / validation | Failures, retries and missing costs |
| --- | --- | --- | --- |
| B4/B5 | External gradient writes, inherited B4 states and trained controller | Model reads and dispatch; external XOR/writeback; gates and causal tests | V4 failures and expanded V5 phase caps above remain part of the cost; full normalized timing/energy not established |
| C35–C37 | Offline geometry and preparation correction; C35 12 geometry-prior + 36 calibration states | Frozen proposals then external commit and measurement; C36 reports 80 parent record/proposal audits and 8 physical replays; C37 4 parent replays | Logs report no generation retries; completed C36 archive recount is not a new run. Missing full archives prevent complete wall-time/weight audit |
| C52 | 576 new offline Jacobians, 7 fits, 112 solves; 1,873 geometry-phase forward invocations | 2,449 candidate-phase forwards, 1,224 metric evaluations and assignments | Four failures remain; 17 path samples per state are dependent observations |
| C54 | Reuses geometry; 7 fits / 7 solves; no new Jacobians, solver retries or search reported | 2,449 candidate-phase forwards, 1,224 metric evaluations and assignments | Four failures remain; zero rescued passes. Inherited preparation/teacher cost is not zero |
| C55 | Recorded read-only arithmetic audit; no fitting or model load | No new model forwards; hash I/O, path reconstruction and equation checks have CPU cost | Does not change C54 gates or decision; audit completion is not LC2 success |

C52–C55 numbers come from their linked original summaries. Phase call counts
include measurement, not only writing. Do not compare them as isolated write
latency. Before a cost claim, record hardware, wall/GPU/CPU time, peak memory,
storage, training/geometry/teacher work, failed attempts, retries, online proposal,
commitment, validation and amortisation separately. Unavailable quantities are
**unknown**, not zero; do not silently exclude inherited preparation or failed runs.

## Reproduction and missing materials

[Read-only checks and A100 inputs](../README.md#verify-existing-evidence-without-a-gpu)
are separate workflows. C35 requires a compatible completed C34 result; C36 a
passed C35 result; C37 a passed C36 FIX1 result with pinned freeze identity.
Their full parent chain, results and frozen tensors remain missing from the
published evidence. C52/C54/C55 full local ZIPs are hash-identified in provenance,
but not published by this change; compact records cannot replace parent ZIPs,
source or tensors for replay. C38–C41 public attachments are also pending.
Do not use a source/update ZIP or a compact extraction as a resume result.

The historical paper and its L1–L5 limitations remain intact. No original SUMMARY,
source, acceptance threshold or historical decision is retroactively changed.

