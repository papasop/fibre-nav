# Line C: Learned Parameter-Update Proposals

**Evidence snapshot: 12 October 2026.** This branch asks whether a learned writer
can propose a selective parameter-memory update in one head forward pass, and
whether that proposal satisfies the same finite content and preservation gates.
It complements B5 operation selection; these writer experiments do not execute
or evaluate the B5 controller.

## What the current evidence adds

The useful result is a separation between **content writing**, **response
preservation**, and **online repair**. A frozen head can produce the requested
content change on local preparation states, yet still violate the protected
response budget. External response correction succeeds in C9. C10 distils
training-state corrected endpoints into the head and removes correction during
writing; acceptance improves relative to the frozen C9 head on the same C10
states, but the complete protocol fails. The negative result is retained.

| Protocol | Training-state result | Development evaluation | Interpretation |
| --- | --- | --- | --- |
| [C7 functional-gate comparison](../evidence/gpt2_c7_functional_gate_a100_v1/) | Both paired arms pass 27/27 after external correction | Both pass 0/9 excluded-input episodes | Three training-qualified folds only; `xor_10` does not qualify. This is a development comparison, not four-input generalisation. |
| [C8 extrapolation diagnosis](../evidence/gpt2_c8_extrapolation_diagnosis_a100_v1/) | Rechecks 120 saved candidates at raw and corrected endpoints | Paired corrected excluded candidates satisfy the response gate, but 0/9 achieve output flips | Restoring protected responses does not recover missing content changes. Diagnosis introduces no new writer training. |
| [C9 local-state generalisation](../evidence/gpt2_c9_state_generalisation_a100_v1/) | Raw 0/12; externally corrected 12/12 | Raw 0/8; externally corrected 8/8 new local states | All four input classes are now training classes. This tests local preparation-state robustness with external correction. |
| [C10 offline correction distillation](../evidence/gpt2_c10_correction_distillation_a100_v1/) | Final raw writer 9/12 | Final raw writer 2/8 new local states; frozen C9 baseline 0/8 on these same states | No online write Jacobian, SVD or response correction. Official decision remains `C10_RAW_TRAINING_FUNCTION_FAILURE`. |

C9 and C10 use different new preparation seeds. Their headline fractions are
not a matched comparison across protocols. C10's two heads are compared on the
same saved states: both produce the requested bits, positive read margins,
required output flips and preservation of the other three cells, while the
response gate distinguishes acceptance. The final C10 writer passes the KL
gate on all 20 evaluated states. Its new-state full passes are both `xor_11`
states; the other three input classes contribute zero full passes.

## Code worth inspecting

Use C10's [archived source](../evidence/gpt2_c10_correction_distillation_a100_v1/source/)
for the latest implementation, and C9 for the correction-assisted predecessor:

| File | Role |
| --- | --- |
| `proposer.py` | Width-32, two-layer `WriterHead`; training-only normalisation; one head call produces a 36,864-coordinate increment from 3,077 features. |
| `functional_writer.py` | Differentiable functional model calls with temporary LoRA-B overrides during offline head training. The frozen GPT-2 base and LoRA-A are not trained. |
| `geometry.py` | External numerical SVD kernel projection and regularised response correction. Numerical rank is not a mathematical feasibility certificate. |
| `worker_c9.py` / `worker_c10.py` | Fixed training, freeze, local-state preparation, actual model evaluation and raw/corrected accounting. C10 rejects Jacobian/correction calls outside the preparation phase. |
| `distillation_loss.py` | C10's offline response-aware objective; supervised targets are C9 corrected **training** endpoints only. |
| `c9_protocol.py` / `c10_protocol.py` | Fixed denominators, gates, exact parent/source identities, saved optimiser progress and outcome decisions. |

The head consumes four 768-dimensional hidden states, four readout margins, and
the **requested output bit**. It predicts an update; it does not infer XOR by
itself. Feature extraction, offline training, parameter submission and finite
acceptance evaluation remain external runtime services.

## Fixed experiment and acceptance scope

The model is `openai-community/gpt2` at revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, with memory/model seed 82931.
Only LoRA-B coordinates in the fixed rank-8 `c_attn` chart of blocks 10 and 11
can change during memory operations. The base model, LoRA-A, access prompts,
read tokens and chart specification remain fixed.

Acceptance requires the requested four-cell content, read margins at least
0.05, an actual output flip, retention of the three unselected cells, protected
response drift at most 0.002, and mean anchor KL at most 0.005. The protected
response averages six selected final-token logits over twelve anchors. KL is
also averaged. **These gates do not establish per-anchor preservation, global
behavioural invariance, or absence of task-level forgetting.** C8 retains
per-anchor values as diagnostics, without changing historical acceptance gates.

C9 trains on twelve previously inspected preparations: all four XOR input
classes, three variants each. Its eight new states are fixed local perturbations
generated after the final head freezes. C10 reuses that frozen head and its
normalisation, distils corrected C9 training endpoints for 1,000 steps, then
performs 360 fixed functional-refinement steps. Its eight new states use another
predeclared seed set. Offline state preparation still uses external geometry.
No new-state score selects a checkpoint, replacement state or teacher fallback.

## Read-only verification

From the repository root, with Python 3.11+ and NumPy:

```bash
python3 scripts/verify_line_c_evidence.py
python3 scripts/verify_line_c_evidence.py --protocol-tests
```

The first command checks exported file hashes, original member identities,
recorded finite gates, summary denominators, actual frozen weight hashes and
NumPy replay against the saved raw parameter states. The second additionally
runs the archived CPU protocol suites in separate subprocesses. Neither command
downloads GPT-2, re-evaluates model responses, or establishes an independent
replication. A passing verifier can and should coexist with C10's failed
experimental decision.

Frozen C9 and C10 heads, training normalisation and twenty saved-state replay
inputs per protocol are exported as NPZ with `allow_pickle=False`. C10 also
exports its MSE-only intermediate head for inspection; **that head has no
standalone archived endpoint acceptance result and is not selected as a winner**.
NPZ exports omit optimiser state and cannot resume an experiment.

For an optional external-head-only proposal:

```bash
python3 scripts/numpy_writer.py --head evidence/gpt2_c10_correction_distillation_a100_v1/weights/final_head.npz --features features.npy --out delta.npy
```

`features.npy` must contain the declared 3,077-dimensional features. This command
does not commit parameters or evaluate preservation/content gates.

## A100 execution and full result artifacts

**Full result attachments: publication pending.** The compact repository does not provide a public download for the full parent/resume archives.

Each evidence directory includes its original one-cell Colab launcher and the
complete archived source dependency closure. Use A100, float32 and disabled
TF32 as declared by the launcher. Supply a **full result ZIP**, not this GitHub
update ZIP, a source ZIP, an NPZ export or the `recorded/` directory.

| New protocol | Required parent result | Resume input |
| --- | --- | --- |
| C7 | Exact compatible full C6 result | Matching full C7 result |
| C8 | Exact compatible full C7 result | Matching full C8 result |
| C9 | Pinned full C8 result | Matching full C9 result |
| C10 | Pinned full C9 result | Matching full C10 result |

The [artifact index](../evidence/line_c_index.json) records complete result
filenames, byte sizes and SHA-256 identities. The compact repository exports
exclude nested parent ZIPs and PT resume checkpoints. Full result files must be
published separately as release assets or in a persistent research archive
before end-to-end reproduction can be described as publicly available. Do not
invent download URLs or mark missing artifacts as published. Uploading a
completed result for resume is a consistency check, not a fresh experiment.

## What remains open

The evidence does not establish native or internally executed parameter
updates, standalone forward-only neural computation, unseen XOR-class
generalisation, independent model seeds, broader prompts, arbitrary program
composition, capacity scaling or lifelong learning. C9's external correction
and C10's partial raw acceptance are different achievements with different
limitations. The next diagnostic is a matched replay of the saved C10 MSE-only
and final functional heads on the same states; it must leave C10's original
decision unchanged.
