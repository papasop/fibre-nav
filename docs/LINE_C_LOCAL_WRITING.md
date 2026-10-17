# Line C: learnable, transferable parameter writing

Evidence snapshot: 16 October 2026. This page records local development progress
in the existing GPT-2 restricted LoRA-B chart. It does not change the scope of
earlier B0–B5 or response-fibre experiments.

For the current capability status, use the [unified evidence matrix](PARAMETRIC_MEMORY_EVIDENCE_MATRIX.md).
[C52/C54/C55 records](../evidence/gpt2_lc2_c52_c54_c55/) extend the development
index: historical seen-class failures and a completed read-only audit, not LC2
acceptance. The [B5/frozen-writer integration](B5_FROZEN_WRITER_INTEGRATION.md)
is separately proposed and has not run.

## Research frontier and subsequent local records

The question is whether a frozen, single-proposal writer can change requested
contents within declared response budgets and transfer beyond its fitting states
and requests. B4 supplies the causal foundation; C35–C37 support local feasibility.
Residual prediction is one mechanism under investigation, not the capability goal.

On 17 October 2026, the following local result ZIP summaries were inspected.
They are **not yet published as repository evidence**; exact source identities
are recorded below. These are recorded outcomes, not independently reproduced
model measurements. The C35–C37 public evidence tier remains unchanged.

| Experiment | Recorded outcome | Interpretation |
| --- | --- | --- |
| C38 excluded input | `C38_SINGLE_FOLD_EXCLUDED_INPUT_FAILURE`; all three arms 0/8 on excluded `xor_10` | Single fold; exclusion from this fit, not historically unseen to the programme. |
| C39 training-only attribution | `C39_RESIDUAL_PREDICTOR_FAILURE_SUPPORTED_TRAINING_ONLY`; raw 12/12, original composite 0/12, measured-residual oracle 12/12 | Supports residual-prediction failure in the training diagnostic; the oracle uses postwrite response and is not a one-shot candidate. |
| C40 finite residual calibration | `C40_SEEN_CLASS_DEVELOPMENT_FAILURE`; raw and new compensation both 10/12, old compensation 0/12, oracle 12/12 | New preparation seeds in fitted classes; no pass-count gain, no excluded-class evaluation. |
| C41 expanded residual fit | `C41_FRESH_SEEN_CLASS_MODULE_FAILURE`; raw and frozen C40 compensation both 21/24, expanded C41 compensation 18/24, oracle 24/24 | Completed result, not pending. Expanded compensation rescues one failure but destroys four passes against either baseline. |

C41 fits on 84 states (12 original plus 72 expanded) before its 24 confirmation
states. Its maximum residual-prediction error decreases relative to the C40 head
on those same states, but endpoint acceptance worsens. All three fitted classes
remain in scope; no excluded-class states are evaluated. These are same-model/chart
experiments. C39–C41 do not close LC2 or reclassify C38. Oracle success is diagnostic,
not evidence of frozen one-shot generalisation or native parameter updates.

### Local archive provenance (publication pending)

The ZIPs below were read from the supplied workspace's Downloads directory.
No public download link is available here. C39/C40/C41 top-level manifest hashes
were checked (94/151/567 members); C38 entries below identify its recovered
diagnostic summary only, without claiming a full recovered-package audit.
No tensors were deserialised and no model was executed. Full public evidence
import, endpoint-level audit and reproducible launchers for these stages remain
separate work; the runnable repository entry points below still cover C35–C37.

| Stage | Local result filename | ZIP SHA-256 | Summary member | Summary SHA-256 |
| --- | --- | --- | --- | --- |
| C38 | `gpt2_lc2_c38_readonly_diagnosis_recovered_results.zip` | `1845488533855f8a002d7a2da2f02a7a3197a9fc3a16ef321944bc7b67ed6ead` | `original_diagnostics/PARENT_SUMMARY.json` | `5ad020aead9ff5b82f6086eb5894b2e2932e74c566c8b5d1b832d9ada9ca8a59` |
| C39 | `gpt2_lc2_c39_training_residual_attribution_results.zip` | `e2b6970100e7e8fd9a2e2afa8adac59f7c075105d4872b3ce549ace4b310eb54` | `SUMMARY.json` | `d7251be00856c325147679326e2cac56df4ecf6d9d2e65aa908d6eb74dcfebcb` |
| C40 | `gpt2_lc2_c40_finite_residual_calibration_results.zip` | `8ee850247e58b86e37b8ca474d35adf39170747cb18d1ce75dcca9718981fb75` | `SUMMARY.json` | `00c9b140610b07082d59cdfff8d396d72327fe7b65c62e12b8fde5de5c14449c` |
| C41 | `gpt2_lc2_c41_expanded_residual_results.zip` | `649a5f71818615298af44820626df914e072de11d7deb75ee325e064c5678fda` | `SUMMARY.json` | `bfc3341f25672d3d36d2e124434077a053d82ca02c5161f2fb883f6c7c473ee0` |

## Frozen local writing: C35–C37 development evidence

**C35–C37 local result:** two frozen composite writers pass all declared gates
on C35's 16 new local states, C36's 32 new local states and C37's 32 states
around changed preparation centres. These are same-model, same-chart development
cohorts. C37 uses previously inspected C36 preparation states as centres; the
cohorts are related and are not independent replications.

| New-state cohort | Raw core | Fixed geometry (`geometry6`) | Learned calibration (`selected6`) |
| --- | --- | --- | --- |
| C35: original local centres | 3/16 | 16/16 | 16/16 |
| C36: frozen local replication | 5/32 | 32/32 | 32/32 |
| C37: changed preparation centres | 7/32 | 32/32 | 32/32 |

Each composite writer produces one final proposal from prewrite information,
followed by external parameter commitment and forward verification. Online
writing uses no Jacobian construction, SVD, postwrite response correction,
backtracking or teacher fallback. Fixed compensation operators are constructed
offline; the learned branch additionally uses an affine calibration. The
uncompensated learned core does not attain complete acceptance. Since the fixed
geometry branch also passes, these results do not establish that learned
calibration is necessary or superior.

**LC1 status:** the declared local development acceptance target is met by these
composite writers; independent confirmation remains pending. LC2 excluded-input
generalisation remains open. Native parameter updates, scaling and continual
learning are not established.

**Evidence tier:** the new records here are extracted from user-supplied execution
logs and accompanied by source code. Full C35–C37 result ZIPs and actual frozen
weight tensors have not been independently inspected in this update. Log
consistency checks are not a rerun of GPT-2 or a weight-identity audit.

[Results, gates and provenance](LINE_C_LOCAL_WRITING.md) ·
[LC1–LC6 validation targets](LINE_C_VALIDATION_TARGETS.md)

## What is frozen and what runs online

The C25/C27 learned writer core is combined with a fixed six-dimensional affine
response-compensation operator. `geometry6` uses the offline geometric predictor;
`selected6` adds a training-selected affine residual calibration. C35 uses 12
fixed geometry-prior states and 36 other calibration states. Both branches are
frozen before generating the new states and are reported in parallel; there is
no per-test choice of the better branch.

One core forward and fixed arithmetic produce a sealed final parameter point
for each branch before any written endpoint is measured. The runtime commits
that point once and verifies it. It still performs tensor assignment and model
evaluation. Offline autograd, SVD, geometric fitting and preparation correction
remain part of the pipeline. Requested output bits are supplied as inputs;
these runs do not test B5 controller dispatch or autonomous XOR computation.

## Acceptance and measured response drift

All seven gates must pass: correct contents, required output flip, unselected
contents retained, protected aggregate response drift at most 0.002, mean anchor
KL at most 0.005, exact read margins at least 0.05 and development read margins
at least 0.05. These protect the declared aggregate response and mean KL;
they do not certify every anchor or unaudited capability.

| New cohort | Raw maximum drift | Geometry maximum drift | Learned-calibration maximum drift |
| --- | ---: | ---: | ---: |
| C35 (16) | 0.0073623657 | 0.0001220703 | 0.0001754761 |
| C36 (32) | 0.0092391968 | 0.0001525879 | 0.0001525879 |
| C37 (32) | 0.0096282959 | 0.0002441406 | 0.0002899170 |

All requested new states have valid preparations in the reported records; no
replacement states or generation retries are reported. Both composite branches
pass every gate on every new state. The raw core passes all non-response gates
but fails the response gate on many states. The numerical margins above are
finite-state measurements, not certified bounds over a neighbourhood.

The reported completed C36 decision is
[`C36_BOTH_FROZEN_WRITERS_LOCAL_REPLICATION_PASS`](../evidence/gpt2_line_c_c35_c37/reported/c36_summary_from_log.json).
Execution is not pending in this log-supported status; full archive/weight audit
remains pending. C35 and C37 original log-extracted summaries are available in
the [reported record directory](../evidence/gpt2_line_c_c35_c37/reported/).

## Cohorts, dependence and historical context

C35 reports 48/48 training and 16/16 previously inspected development acceptance
for each composite arm; its raw core passes 23/48 and 10/16. New states number
four per XOR class in C35 and eight per class in C36/C37. C36 reports an audit of
80 parent records, exact reproduction of 80 proposals and eight physical
replays. The supplied C36 transcript resumes a completed archive and recounts
its records; this recount is not another experiment.

C37 fixes four prewrite C36 preparations per class by index, then creates two
new perturbations per centre. The 16 centres were inspected in C36 but were not
used to refit the deployed writers. Its 32 new states transfer the local centre
within the same four trained input classes. The C37 log reports four physical
parent replays. It does not supply independent model or chart replication.

Earlier results remain important: C9 achieves 0/8 raw and 8/8 externally corrected
acceptance; C10 reports 9/12 training and 2/8 inspected-state raw acceptance;
C27 obtains 76/76 training acceptance but 6/16 inspected-state acceptance.
These historical counts are transcribed from earlier conversation reports and
are not newly audited by this package. C35–C37 deploy a different composite
proposal mechanism; they do not retroactively turn earlier raw failures into
passes. See the original archives before using historical counts as primary
evidence.

## Provenance and reproduction

The [evidence folder](../evidence/gpt2_line_c_c35_c37/) contains verbatim supplied
logs, exact extracted summary objects, their SHA-256 identities and a read-only
consistency verifier. These JSON objects are named `summary_from_log`, not
archived `SUMMARY.json`, to avoid implying independent result-ZIP verification.
No real result ZIP, teacher tensors or writer checkpoint is distributed here.

The [source and launcher folder](../external_tests/gpt2_line_c_c35_c37/) includes
the original C35, C36 FIX1 and C37 single-cell launchers, and the complete C37
source tree with inherited C36/C35 sources. Synthetic tests do not establish
real-model acceptance. Reproduction requires the original compatible parent
result archives; source ZIPs cannot substitute for result ZIPs.

## What this supports

LC1 local development acceptance can be recorded as achieved by a frozen
composite writer under the declared model/chart/access protocol. The fixed
geometry control's success is substantive: learning the residual calibration
is not shown necessary. Generalisation outside these local preparation
distributions, untouched input-class holdouts, independent model seeds, long
operation sequences, capacity and cost scaling, continual learning and internal
parameter updates require separate evidence.
