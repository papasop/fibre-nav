# Qwen MFI operation interface — L1 development

This branch implements the requested chain:

`MFIController -> WRITE/OVERWRITE -> Qwen LoRA-B chart -> response/KL acceptance`

Unlike the separate `agent/` SQLite prototype, the bit is read from modified
Qwen logits. No database supplies the returned value. The controller accepts
explicit operations from the caller; Qwen does not parse the commands itself.

## First real run

`results/l1_development_001/report.json` records a successful single-seed CPU run:

| State | Read bit | Target margin | Response drift | Anchor KL |
| --- | --- | --- | --- | --- |
| Initial | 0 | 0.8231 (for 0) | 0 | 0 |
| WRITE 1 | 1 | 1.4518 | 0.008450 | 0.008295 |
| OVERWRITE 0 | 0 | 1.4403 | 0.019582 | 0.001457 |

Frozen budgets: response infinity norm <= 0.02; average full-vocabulary
next-token KL <= 0.01; target-vs-other bit logit margin >= 1.0.
Both references remain fixed at the original state through overwrite. Runtime
was 27.32 seconds excluding setup. This is 1/1 development run, not confirmation.

## Mechanism and scope

Freeze Qwen3-0.6B at the declared immutable revision. In the last two layers,
wrap q_proj/v_proj with rank-four LoRA, fixed seeded random A, zero-initialized
trainable B and scaling 2. Only the 24,576 B parameters change. The local operator
is `v - J.T @ solve(J @ J.T + 1e-8 I, J @ v)`, computed in float64, following the
repository R23d Jacobian/near-kernel approach. This is a ridge near-kernel operator,
not an exact projector. The implementation is independent of PEFT.

Three fixed plain-text anchors define three A-minus-B logit responses. Each
candidate is checked against the original response and full-distribution KL.
Backtracking also requires improved target margin. The response/KL bounds apply
to these anchors and next-token distributions only, not general Qwen behavior.

The memory prompt and two candidate tokens are frozen in the protocol. READ is
argmax over those two tokens, not full-vocabulary top-1 generation, conversational
recall or held-out-expression access. The first target is the opposite of the
initially read bit, preventing an unchanged baseline from passing WRITE.

An operation commits only when the final margin and finite budgets pass. Failure
or exception restores the entire pre-operation B vector. OVERWRITE is disallowed
until WRITE succeeds. Tests cover accepted writes, failed-operation rollback,
exception rollback, non-finite rejection and unsupported operations.

## Run

Use a Python environment with torch 2.8.0+cpu, transformers 4.56.2 and safetensors
(the complete observed versions are recorded in report.json):

```bash
python -m unittest discover -s fibre-qwen/mfi/tests -v
OMP_NUM_THREADS=4 python fibre-qwen/mfi/qwen_l1.py --out /path/to/new-run-directory
```

A new output directory is required. It contains a protocol written before model
execution, report and final `chart.safetensors` with A and B tensors. The tensor
file is a custom chart checkpoint, not a PEFT adapter; use the exact source,
model revision, target modules and scale recorded here to interpret it.
No paid hardware is provisioned. Dependencies and model downloads must be available.

The frozen source snapshots next to the report match its code hashes. Original
R23d is at `fibre-qwen/protocols/r23d/fibre_qwen_r23d.py`; its training-loss protocol
and prior evidence remain distinct from this new binary memory experiment.

## Claim boundary and next gate

L1 interface preflight only: no held-out expressions (L2), reusable compositional
address atlas (L3), MOVE/RETURN (L4), multiple selectively operated cells (L5),
causal superiority over ordinary editing, independent seeds, or autonomous agency.
No no-move/random controls were run, so this demonstrates execution and finite
acceptance, not a comparative claim about why the method works.

Before scaling: freeze unused seeds, add no-move and matched-norm controls, verify
checkpoint reload, and test excluded expressions. Keep this result separate from
any later repair or expanded protocol. SQL memory may store audit metadata but
must never stand in for the neural READ operation.


## Checkpoint reload verified

[Fresh-process verification](results/l1_reload_001/README.md) passed all 7 gates:
exact B restoration, original margin/response/KL reproduction, nonzero difference
from initial parameters, correct bit, initialized controller, and rejection of an
incorrect declared bit without changing committed state. Six engineering tests pass.

```bash
python fibre-qwen/mfi/verify_reload.py --source fibre-qwen/mfi/results/l1_development_001 --out /path/to/new-reload-report
```

`checkpoint.restore` verifies caller-supplied SHA-256, exact metadata, tensor keys,
shapes, dtypes and finiteness; fixed A must match the constructed chart. It applies B
transactionally and returns an initialized MFIController only after read/margin and
finite response/KL gates pass. References come from the freshly constructed original
model, not from the loaded endpoint. This supports the exact declared fixed-A chart;
it is not a generic PEFT loader or a proof that source reports are authentic.

The original final checkpoint has now been reloaded, but WRITE-1 intermediate
persistence, independent seeds, matched controls and L2 remain open.
