# B5 controller with a frozen writer: proposed integration acceptance

**Status: specification only; execution pending.** C36/C37 did not test controller
dispatch. B5 operation selection and LC1 frozen-writer results are separate
experiments; neither their juxtaposition nor passing software tests establishes
an integrated end-to-end result. This document starts no GPU work and does not
assign a new historical experiment number.

## Frozen contract before execution

Record exact controller, writer, geometry, normalization, adapter and parent-state
hashes; pin model/tokenizer, chart coordinates, access prompts, action grammar,
routing and evaluation code. Check compatibility between the B5 and LC1 charts
and response definitions before claiming integration. An incompatible interface
requires a newly declared development protocol, not an implicit conversion.
Declare all input states, seeds, splits, budgets, step limits, branch selection
and controls before inspecting outcomes. Freeze both the writer and controller
before the accepted run; any fitting/adaptation is a separately labelled phase.
Report fixed-geometry and learned-calibration writers separately without selecting
a branch using test outcomes. Retain all requested preparations and failures.

| Component | Responsibility |
| --- | --- |
| Controller model | Select grammar-constrained READ, XOR, WRITE and HALT operations and addresses. |
| Memory model | Produce actual parameter-dependent readouts through fixed access prompts. |
| Frozen writer | Produce one sealed update proposal from allowed prewrite state/features and the requested output bit; no online refit or corrective loop. |
| External runtime | Route adapters, execute XOR on the actual read values, supply the requested bit, commit the parameter update and record the trace. |
| Evaluator | Measure endpoint gates and controls; keep postwrite measurements out of the proposal path. |

The proposed experiment does not internalise XOR or parameter commitment and
cannot establish LC6. Offline geometry/preparation remains external. No-op
symbolic readiness is only a software preflight, not a physical-memory result.

## Required physical episodes and gates

Predeclare the four XOR inputs and initially wrong output cell, with all non-output
contents specified. Read both inputs from the actual memory state, execute the
model-selected sequence, pass the runtime XOR result to the frozen writer, commit
once, and read the actual resulting parameter state. Record the prewrite state,
actions, addresses, read values/margins, computed bit, proposal, committed state,
endpoint observations and failure reason for every requested episode.

Acceptance requires all of the following for each declared positive episode:

- Correct operation sequence and HALT within the fixed action limit; no runtime
  error, hidden teacher fallback, refitting, branch switching or postwrite correction.
- Actual input reads match prepared contents; XOR uses these reads; WRITE reaches
  the declared output address; the output flips to the correct requested value.
- Every unselected cell retains its specified content; all exact and development
  read margins meet `0.05`; mean anchor KL is at most `0.005`; the declared
  aggregate protected-response drift is at most `0.002`.
- Use the complete source protocol's response reduction, anchors, precision and
  seven endpoint gates, including bits, flip and retention. Equal numeric limits
  alone do not make different response maps interchangeable. Resolve and freeze
  any mapping before execution; report both source gate sets if necessary.
- Verify frozen controller/writer identities and the declared controller probes;
  zero drift on fixed probes only covers those probes, not all model behaviour.

Acceptance thresholds are inherited as a proposed contract, not edits to existing
records. A missing/invalid preparation counts in the requested denominator;
report invalid, runtime-failed and gate-failed cases separately. Do not silently
replace states, relax gates, extend compute or resubmit a rejected proposal. Any
continuation is a new, linked development attempt with its actual extra budget.

## Causal controls and reporting

Include read-value substitution, stored-input substitution, reset and restore
from explicit saved states, following the B4 causal distinction. Specify expected
changes in the requested bit and actual output for each intervention before the
run. For reset, separately require that it destroys the intended stored contents
and expected computation; a reset control need not pass the positive output-flip
gate. Keep any original `pass=false` field and report control acceptance separately.
Restoration must recover the pinned state and its predicted reads/behaviour.
A skipped-WRITE or no-op control must fail the required output flip when the
output was initialized incorrectly. These controls must run within the integrated
system; inherited B4/V3 controls cannot substitute for them.

Archive every episode and intervention, raw gate flags, rejected proposals,
checkpoint identities, failure traces and full provenance. Report offline geometry,
preparation, fitting, controller training, retries, online proposal/commit and
measurement costs separately, including inherited development costs. Separate
software preflight, archive audit and actual GPU execution in the final decision.

An integrated development pass would establish only the declared B5-plus-frozen-
writer system. LC2 still requires requests excluded from all fitting/selection;
LC3–LC6 each require their own evidence. Plan fresh memory/chart seeds and unseen
request evaluations separately; never pool related cohorts as independent trials.

[Unified evidence matrix](PARAMETRIC_MEMORY_EVIDENCE_MATRIX.md) ·
[LC1–LC6 validation targets](LINE_C_VALIDATION_TARGETS.md)
