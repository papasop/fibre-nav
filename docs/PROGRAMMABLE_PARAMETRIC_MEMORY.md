# Programmability of Parametric Memory

A research framework for fixed-access parametric memory, with selective updates, declared response-preservation budgets, and causal read–compute–write tests.

[Repository overview](../README.md) · [B5 V5 complete XOR](../evidence/gpt2_b5_xor_continuation_a100_v5/) · [B5 V3 archive](../evidence/gpt2_b5_single_backbone_a100_v3/) · [B5 V2 archive](../evidence/gpt2_b5_trained_dispatch_a100_v2/) · [B4 causal archive](../evidence/gpt2_b4_minimal_causal_a100_v1/)

## Current frontier: learnable, transferable parameter writing

B4 is the causal foundation; Line C is the current research frontier. The
question is whether a frozen writer can propose a single parameter update that
changes requested contents within declared response budgets and transfers beyond
its fitting states and requests. Local feasibility and transfer are separate claims.

[The Line C results page](LINE_C_LOCAL_WRITING.md) holds the full C35–C37 tables,
gates, provenance and limitations. Both fixed-geometry and learned-calibration
composites pass the reported local cohorts; this does not establish a benefit
from learned calibration. Evidence is log-derived; full result/weight audit and
independent confirmation remain pending. [LC2 excluded-request generalisation
and the other validation targets](LINE_C_VALIDATION_TARGETS.md) remain open.

The [subsequent local C38–C41 summaries](LINE_C_LOCAL_WRITING.md#research-frontier-and-subsequent-local-records)
retain excluded-input and residual-module failures, including the completed C41
result; public archive import remains pending.

The sections below retain the archived selective-write and causal-operation
foundations and their version history. For current frozen-writer reproduction,
see the [Line C launchers and required parent results](../external_tests/gpt2_line_c_c35_c37/).

## Research question and contributions

Can fixed read rules expose selectively writable cells in a shared neural
parameter state, preserve declared responses within budgets, and support causal
read–compute–write operations? B0–B3 supply the selective-write and addressing
foundations; B4 supplies causal evidence; B5 adds learned operation selection.
Line C asks whether the writing mechanism itself can be learned, frozen and
transferred, with each step assessed against separate acceptance targets.

The contributions are the operational interface, protocol-specific acceptance
criteria, causal tests and combination of evidence. A memory read is a model
forward pass through the declared access prompts. Writing changes restricted
LoRA-B coordinates while base weights and LoRA-A remain frozen. Protected
responses are finite declared measurements, not the entire model's behavior.

## Strongest experimental evidence

### B4 minimal causal read–compute–write

The [original summary](../evidence/gpt2_b4_minimal_causal_a100_v1/recorded/SUMMARY.json)
records `B4_MINIMAL_CAUSAL_DEVELOPMENT_PASS` for development seed `82931` and
fixed input `01`. The [causal traces and checks](../evidence/gpt2_b4_minimal_causal_a100_v1/recorded/seeds/82931/cases.json)
and [endpoint records](../evidence/gpt2_b4_minimal_causal_a100_v1/recorded/seeds/82931/records.json)
separate model reads, external deterministic XOR and numerical parameter writes.
Bits below are `[A, B, sentinel, output]`.

| Intervention / stage | Recorded state | Causal observation |
| --- | --- | --- |
| Prepare | `[0,1,1,0]` | Output starts opposite the correct XOR value. |
| Normal writeback | `[0,1,1,1]` | Actual reads `01`; external XOR is `1`; output flips `0→1`. |
| Wrong-read writeback | `[0,1,1,0]` | From the saved normal endpoint, override the read channel with `11`; external XOR becomes `0` and actual output flips `1→0`. |
| Stored-input substitution | `[1,1,1,1]` | Separately restart from the normal endpoint; change stored A while retaining B, sentinel and stale output. |
| Substituted writeback | `[1,1,1,0]` | Actual reads now return `11`; XOR is `0`; output flips `1→0`. |
| Reset / restore | Reset reads `00`; restore reads `01` | Resetting LoRA-B changes the computed result to `0`; reloading the normal endpoint restores result `1`. |

All five endpoints pass the archived development read-margin and averaged
response/KL gates; the archive records 19 causal/endpoint checks. The declared
limits are minimum read margin `0.05`, averaged protected-response drift `0.002`
and mean anchor KL `0.005`. These apply to the protocol's six response coordinates
averaged over its anchors and mean KL, not per-prompt or global preservation.
The reused V2 expression family is auxiliary development data, not fresh
acceptance evidence. The conclusion is single-input, single-development-seed
causal mechanism evidence; full truth-table and independent-seed confirmation
remain open.

### B5 V2 two-model development

Start with the [B5 V2 evidence directory](../evidence/gpt2_b5_trained_dispatch_a100_v2/).
Its [summary](../evidence/gpt2_b5_trained_dispatch_a100_v2/recorded/SUMMARY.json)
records `B5_MODEL_DISPATCH_DEVELOPMENT_PASS`. A separately trained GPT-2
controller selects grammar-constrained READ, XOR, WRITE and HALT actions for a
second GPT-2 memory model. The controller uses seed `82932`; physical memory
states reuse B4 development seed `82931`. This is a two-model system, not
single-model self-modification.

The [training split](../evidence/gpt2_b5_trained_dispatch_a100_v2/recorded/controller_split.json)
and [readiness record](../evidence/gpt2_b5_trained_dispatch_a100_v2/recorded/controller_readiness.json)
show **8/8 symbolic-runtime readiness episodes: two address combinations excluded
from controller training × four XOR inputs**. These are symbolic feedback runs,
not eight actual parameter-memory experiments or four independently prepared
physical XOR inputs.

| Actual memory case | Start → final state | Recorded acceptance |
| --- | --- | --- |
| [normal](../evidence/gpt2_b5_trained_dispatch_a100_v2/recorded/normal/COMPLETE.json) | `[0,1,1,0] → [0,1,1,1]` | Positive case passes. |
| [stored_substitution](../evidence/gpt2_b5_trained_dispatch_a100_v2/recorded/stored_substitution/COMPLETE.json) | `[1,1,1,1] → [1,1,1,0]` | Positive case passes. |
| [restored](../evidence/gpt2_b5_trained_dispatch_a100_v2/recorded/restored/COMPLETE.json) | `[0,1,1,0] → [0,1,1,1]` | Positive case passes. |
| [reset](../evidence/gpt2_b5_trained_dispatch_a100_v2/recorded/reset/COMPLETE.json) | `[0,0,0,0] → [0,0,0,0]` | `pass=false`; only `required_output_flip=false`. |

Reset correctly computes `0 XOR 0`, so no output flip occurs. Its other recorded
checks, including endpoint gates, are true. The [frozen worker's aggregate rule](../evidence/gpt2_b5_trained_dispatch_a100_v2/source/worker_b5.py)
requires the three positive cases plus `reset_changes_read`,
`restore_recovers_read` and `stored_substitution_changes_read`. All three
comparison controls are true. Reset is therefore an accepted causal comparison,
not a successful nontrivial write; its raw `pass=false` remains unchanged.

The controller selects operations; the external runtime computes XOR, enforces
the output-address constraint and performs numerical gradient-based writes.
Reused memory read prompts remain development diagnostics. Independent B5
confirmation, arbitrary programs, native internal gradient execution, lifelong
learning and zero forgetting are not established.

### B5 V3 single-backbone development

The [V3 archive](../evidence/gpt2_b5_single_backbone_a100_v3/) records
`B5_SINGLE_BACKBONE_DEVELOPMENT_PASS` in its
[original summary](../evidence/gpt2_b5_single_backbone_a100_v3/recorded/SUMMARY.json).
The [architecture record](../evidence/gpt2_b5_single_backbone_a100_v3/recorded/ARCHITECTURE.json)
declares one GPT-2 backbone and two independent adapters: `default(memory)` and
`controller`. The external runtime switches between them; this is not
self-modification in one shared update chart. Base weights and both LoRA-A
charts remain frozen. Adapter routing, XOR and numerical gradient writes
remain external services.

The [readiness record](../evidence/gpt2_b5_single_backbone_a100_v3/recorded/controller_readiness.json)
contains 8/8 symbolic-runtime episodes: two training-excluded address tuples
× four XOR inputs. These task combinations were already inspected in V2;
they are development tests, not new independent confirmation. The new controller
seed `82933` does not change the reused B4 memory seed `82931` or its states.

| Actual V3 memory case | Start → final state | Recorded acceptance |
| --- | --- | --- |
| [normal](../evidence/gpt2_b5_single_backbone_a100_v3/recorded/normal/COMPLETE.json) | `[0,1,1,0] → [0,1,1,1]` | Positive case passes. |
| [stored_substitution](../evidence/gpt2_b5_single_backbone_a100_v3/recorded/stored_substitution/COMPLETE.json) | `[1,1,1,1] → [1,1,1,0]` | Positive case passes. |
| [restored](../evidence/gpt2_b5_single_backbone_a100_v3/recorded/restored/COMPLETE.json) | `[0,1,1,0] → [0,1,1,1]` | Positive case passes. |
| [reset](../evidence/gpt2_b5_single_backbone_a100_v3/recorded/reset/COMPLETE.json) | `[0,0,0,0] → [0,0,0,0]` | `pass=false`; only `required_output_flip=false`. |

As in V2, reset is a separate comparison: changed reads and recovery on restore
support dependence on stored contents. The original false positive-operation
flag is retained, not converted into a nontrivial-write pass. Three positive
cases plus the recorded reset/restoration/substitution comparisons determine
the aggregate result in the [V3 worker](../evidence/gpt2_b5_single_backbone_a100_v3/source/worker_b5.py).

The fixed initial-task controller score probe has maximum absolute change `0`
for each of the four recorded memory writes. This measurement covers only the
declared probe. It is consistent with inactive-adapter isolation, not a global
instruction-preservation guarantee or interference-free shared-chart
self-modification. Independent B5 confirmation and Line C update internalisation
remain open.

### B5 V4–V5 complete XOR truth-table development coverage

The [V5 archive](../evidence/gpt2_b5_xor_continuation_a100_v5/) records
`B5_XOR_CONTINUATION_DEVELOPMENT_PASS`, with **4/4 actual XOR inputs** covered
across V4 and V5. Each independently prepared input state has an initially
incorrect output, so every accepted episode requires a physical parameter-memory
writeback and output flip. This is a minimal operational XOR test, not a claim
of Boolean completeness or universal program execution.

The [unchanged V4 parent](../evidence/gpt2_b5_xor_truth_table_a100_v4/)
records **2/4 passes**. Case 10 failed preparation on development paraphrases;
case 11's writeback candidate failed exact/paraphrase margin gates and was rolled
back. V5 inherits V4's passing 00/01 records unchanged. It resumes the actual 10
preparation checkpoint from step 600 to 1200, then writes for 1080 steps; for 11
it restarts writing from the saved passed preparation because the rejected V4
candidate and optimizer had not been retained. V5 retains both candidate and
optimizer checkpoints, including rejected attempts.

| Input | Start → final state | V5 origin | Minimum exact margin | Minimum development-paraphrase margin | Mean anchor KL |
| --- | --- | --- | ---: | ---: | ---: |
| 00 | `[0,0,1,1] → [0,0,1,0]` | Inherited V4 pass | 3.6542 | 3.5944 | 0.003497 |
| 01 | `[0,1,1,0] → [0,1,1,1]` | Inherited V4 pass | 5.0898 | 5.3922 | 0.003140 |
| 10 | `[1,0,1,0] → [1,0,1,1]` | Actual-checkpoint continuation | 12.3999 | 12.6711 | 0.002507 |
| 11 | `[1,1,1,1] → [1,1,1,0]` | Write restart from preparation | 3.3393 | 4.5527 | 0.003643 |

Original thresholds and writer bytes remain unchanged. Final writeback drift is
`1.52587890625e-05` for every case; the averaged response budget is `0.002`, mean
anchor KL budget `0.005`, and minimum exact/paraphrase margin `0.05`.
The raw `heldout` fields refer to previously inspected development paraphrases,
not new evaluation data. [Recorded origin declarations](../evidence/gpt2_b5_xor_continuation_a100_v5/recorded/CONTINUATION_PLAN.json)
and [summary](../evidence/gpt2_b5_xor_continuation_a100_v5/recorded/SUMMARY.json)
make the continuation scope explicit.

The fixed V3 controller (seed 82933) selects the action sequence without additional
training; memory seed 82931 is reused. The external runtime routes separate
adapters, computes XOR and performs numerical updates. The action template was
provided during controller training. Exactly zero controller-probe drift is an
adapter-isolation/routing check, not a shared-chart self-modification result.
Earlier causal controls are retained in the V3 parent and were not rerun for
each truth-table input. Independent seeds, fresh read prompts and expanded causal
controls are the next validation targets.

## Current status

| Stage / line | Archive-supported status | Boundary |
| --- | --- | --- |
| GPT-2 B0–B3 foundations | Reported results within the [paper](../paper/A_Programmable_Address_Space_for_Neural_Memory.pdf) and legacy [L1–L5 evidence ladder](MFI_L1_L5_EVIDENCE_LADDER.md) | Separate local/single-seed audits; not unified multi-seed confirmation. Legacy L5 v2.2.4 is explicitly identified as roadmap B3 by the [B4 parent documentation](../evidence/gpt2_b4_causal_xor_a100_v1/README.md). |
| Earlier B4 XOR V1 | `B4_MINIMAL_DEVELOPMENT_PASS` | Two input states; substituted output was already correct, read blocking only compared controller results, reset-sensitivity Boolean was weak. |
| Stronger B4 minimal causal V1 | `B4_MINIMAL_CAUSAL_DEVELOPMENT_PASS` | One seed and fixed input `01`; stronger executed interventions. |
| Full XOR B4 V2 | `PROTOCOL_READY_RESULTS_PENDING` — [status file](../external_tests/gpt2_b4_full_xor_a100_v2/STATUS.json) | Protocol/source available; no real GPT-2 result claimed. |
| B5 V2 | Two-model learned-dispatch development pass | Symbolic readiness and actual memory runs are different evidence units. |
| B5 V3 | Single-backbone, partitioned-adapter development pass | One backbone; separate runtime-routed adapters; reused memory seed and previously inspected tasks. |
| B5 V4 historical parent | Four-input development failure, 2/4 | Failures are retained; this record is not relabeled as passing. |
| B5 V5 complete XOR | Four-input development coverage passes, 4/4 | Inherits 00/01; continues 10/11; frozen controller and original thresholds; external numerical execution. |
| B5 independent confirmation | Pending | Neither a new controller seed nor adapter isolation establishes independent memory replication. |
| Qwen | Separate lines, detailed below | Do not transfer a Qwen result to GPT-2 memory or vice versa. |
| Line C | Local composite-writer development acceptance reported in C35–C37 | Independent confirmation and native parameter updates remain open. |

The V4/V5 update was prepared against repository revision
`7c19a571e7c17c41ddc0c5ea817cafae8f73de56` and the hash-verified V5 total archive
(SHA-256 `30ac85d7208ba1525b7cc8934ffc94e4d42bb8b593a17bba715db2f652dd66dd`).
Statuses refer to available archives, not filenames or the number of software tests.

### Qwen status and missing version evidence

The current tree does not contain a located V23 development-pass archive,
its fresh-evaluation failure record, or the proposed later V23 development
sequence. Those three statuses cannot be independently established here and
must remain separate when their artifacts become available. They are not
reconstructed from the similarly numbered **R23** optimizer experiments.

The available [R23c-r2 provenance](../fibre-qwen/evidence/r23c_r2/PROVENANCE.md)
describes a one-seed precision/development audit; the separate
[R23d provenance](../fibre-qwen/evidence/r23d/PROVENANCE.md) describes a restricted
Qwen3-0.6B confirmation over training-order seeds. Neither is a V23
parameter-memory confirmation. In the frozen Qwen3-1.7B CPU interface line,
[V8's latest checkpoint](../fibre-qwen/evidence/qwen_cpu_interface_v6_v8/v8/checkpoint_103/README.md)
is paused at 103/128 and incomplete; the [V9 audit](../fibre-qwen/evidence/qwen_cpu_chain_v9/README.md)
reports a passed external two-step interface test. These do not establish
parametric-memory writes or native planning.

## Verify archives without a GPU

Run from the repository root using Python 3; these scripts use the standard
library and do not load model weights or launch training:

```bash
python3 evidence/gpt2_b5_xor_continuation_a100_v5/verify_archive.py
python3 evidence/gpt2_b5_xor_truth_table_a100_v4/verify_archive.py
python3 evidence/gpt2_b5_single_backbone_a100_v3/verify_archive.py
python3 evidence/gpt2_b5_trained_dispatch_a100_v2/verify_archive.py
python3 evidence/gpt2_b4_minimal_causal_a100_v1/verify_archive.py
python3 evidence/gpt2_b4_causal_xor_a100_v1/verify_archive.py
```

The [V5 verifier](../evidence/gpt2_b5_xor_continuation_a100_v5/verify_archive.py)
checks original hashes, all three embedded parents, development gates, traces,
unchanged controller weights, inherited cases and saved candidate storage.
The adjacent V4 verifier preserves and validates the historical 2/4 failure.

The [V3 verifier](../evidence/gpt2_b5_single_backbone_a100_v3/verify_archive.py)
also checks architecture declarations and recorded controller-probe preservation.
The [B5 V2 verifier](../evidence/gpt2_b5_trained_dispatch_a100_v2/verify_archive.py)
checks CRC, manifest hashes, recorded readiness, physical traces and control
consistency. The [stronger B4 verifier](../evidence/gpt2_b4_minimal_causal_a100_v1/verify_archive.py)
checks the original result/source/launcher identities, endpoint records and
recorded causal gates. Successful checks establish archive consistency, not
fresh model execution, independently validated tensor readouts or replication.

The V3 runtime self-tests are separate software checks, not model evidence:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 evidence/gpt2_b5_single_backbone_a100_v3/source/test_dispatch.py
```

Legacy checks remain available:

```bash
python3 verify_mfi_l1_l5.py
python3 verify_evidence.py
```

**Known checkout limitation:** the [L1–L5 verifier](../verify_mfi_l1_l5.py)
currently stops on a missing manifest target,
`evidence/l3_category_v1_9_6_0/evidence/v1_9_6_0/console.log`.
This is not a passing end-to-end archive check. The separate
[response-fibre verifier](../verify_evidence.py) validates available R2/PC1/PC3/PC4
records and reports PC2 blocked because its required report is unavailable.
No missing evidence is reconstructed or acceptance threshold changed here.

## A100 experiments and input files

These are actual experiments, unlike the read-only checks above. Use a Colab
A100 runtime with CUDA PyTorch available; copy the **whole linked PY file** into
one cell. Do not run it during documentation or archive verification.

| Experiment / protocol | One-cell launcher | Input |
| --- | --- | --- |
| `GPT2_B5_XOR_CONTINUATION_A100_V5` | [B5 V5 PY](../evidence/gpt2_b5_xor_continuation_a100_v5/GPT2_B5_XOR_CONTINUATION_A100_V5.py) | Complete failed V4 result for replaying this development continuation, or a matching V5 result for resume. Completed-result resume is not a fresh replication. |
| `GPT2_B5_SINGLE_BACKBONE_A100_V3` | [B5 V3 PY](../evidence/gpt2_b5_single_backbone_a100_v3/GPT2_B5_SINGLE_BACKBONE_A100_V3.py) | Passed minimal-causal B4 result or B5 V2 result to extract its B4 parent and train a new V3 controller; matching V3 results for resume. |
| `GPT2_B5_TRAINED_DISPATCH_A100_V2` | [B5 V2 PY](../evidence/gpt2_b5_trained_dispatch_a100_v2/GPT2_B5_TRAINED_DISPATCH_A100_V2.py) | A passed minimal-causal B4 result ZIP for new controller training, or a compatible B5 V2 result ZIP for resume. |
| `GPT2_B4_MINIMAL_CAUSAL_A100_V1` | [B4 minimal-causal PY](../evidence/gpt2_b4_minimal_causal_a100_v1/GPT2_B4_MINIMAL_CAUSAL_A100_V1.py) | Fresh run starts from zero B; continuation accepts compatible results of this same protocol. |
| `GPT2_B4_FULL_XOR_A100_V2` | [B4 full-XOR V2 PY](../external_tests/gpt2_b4_full_xor_a100_v2/GPT2_B4_FULL_XOR_A100_V2.py) | Fresh run starts from zero B; only compatible full-XOR V2 results resume. Results remain pending. |

For V5, the [input and resume guide](../evidence/gpt2_b5_xor_continuation_a100_v5/)
links the V4 parent and V5 result. Drive backups use the separate folder
`GPT2_B5_XOR_CONTINUATION_A100_V5_RECOVERY`. The source/launcher ZIP is not
a result input. The wrapper repairs a missing dependency-parent directory
without changing the archived worker or resume identity.

For B5 V2, download the actual input artifact:

- **New V2 experiment:** [gpt2_b4_minimal_causal_a100_results.zip](../evidence/gpt2_b4_minimal_causal_a100_v1/gpt2_b4_minimal_causal_a100_results.zip), protocol `GPT2_B4_MINIMAL_CAUSAL_A100_V1`.
- **Resume:** [gpt2_b5_trained_dispatch_a100_results.zip](../evidence/gpt2_b5_trained_dispatch_a100_v2/gpt2_b5_trained_dispatch_a100_results.zip), protocol `GPT2_B5_TRAINED_DISPATCH_A100_V2`. The published completed result is a resume input, not a fresh replication.

**Do not upload `GPT2_B5_TRAINED_DISPATCH_A100_V2.zip` as results:** it is the
source/launcher package. The same distinction applies to the uppercase B4
source ZIPs. Older B4 XOR V1 and full-XOR B4 V2 are not the minimal-causal B4
parent required by this B5 launcher.

The B5 wrapper exposes `INPUT_ZIP` and `USE_GOOGLE_DRIVE`. With an empty input,
the [launcher](../evidence/gpt2_b5_trained_dispatch_a100_v2/source/launch_b5.py)
first checks verified V2 backups in
`MyDrive/GPT2_B5_TRAINED_DISPATCH_A100_V2_RECOVERY`, then prompts for one upload.
Set `INPUT_ZIP` to the intended uploaded result path to choose it explicitly.
It also supports extracting a B4 parent from a
`GPT2_B5_MODEL_DISPATCH_A100_V1` result, retraining the V2 controller rather than
restoring V1 action records; no V1 artifact is supplied as the preferred entry.
Incompatible identities are rejected. B5's XOR and numerical writer remain
external even when operation selection is learned.

For V3, the [single-backbone launcher](../evidence/gpt2_b5_single_backbone_a100_v3/source/launch_b5.py)
uses `MyDrive/GPT2_B5_SINGLE_BACKBONE_A100_V3_RECOVERY` for verified automatic
resume. Supply the B4 result above, or the V2 result to extract its B4 parent;
V2 controller/action records are not continued as V3. Use the
[matching V3 result ZIP](../evidence/gpt2_b5_single_backbone_a100_v3/gpt2_b5_single_backbone_a100_results.zip)
to resume protocol `GPT2_B5_SINGLE_BACKBONE_A100_V3`.
`GPT2_B5_SINGLE_BACKBONE_A100_V3.zip` is a source package, not a result input.
The published completed V3 result can restore a completed run; doing so does
not constitute a new independent experiment.

## Mathematical mechanism

Let \(R(\theta)\) be the declared protected response map in the permitted
parameter chart, with Jacobian \(J=DR(\theta)\). Then \(Jv=0\) means
first-order response preservation, not exact preservation after a finite step.
If columns of \(N\) span the permitted kernel and \(H\) is a differentiable
readout map, then

\[
C=DH(\theta)\,N
\]

describes local readout controllability within kernel directions: for
\(v=N\alpha\), the first-order readout change is \(C\alpha\). Its rank and
conditioning describe local possibilities, not arbitrary finite writes or
unlimited memory capacity.

Finite updates are additionally constrained by curvature, response/KL budgets,
parameter restrictions and numerical error. The archived writer uses a
ridge-regularized near-kernel operator, rather than an exact projector, and
checks finite endpoint gates. Standard null-space projection is background
mathematics. The research contribution is the memory interface, acceptance
protocols, causal tests and combined experimental evidence; no global fibre
bundle, zero-forgetting theorem or new projection theorem is claimed.

## B0–B5 and Line C roadmap

| Milestone | Role | Evidence / next boundary |
| --- | --- | --- |
| B0–B2 foundations | Restricted writable memory, reusable addressing and local transport | Consult the original L1–L4 archives and paper; legacy labels are preserved rather than silently relabeled as new experiments. Standalone L2 results are unavailable. |
| B3 | Selective operations in one shared parameter state | Legacy L5 v2.2.4: WRITE, selective OVERWRITE, SWAP, MOVE and RETURN in the declared chart; separate single-seed scope. |
| B4 | Causal read–compute–write with externally selected operations | Minimal causal development passes; complete truth-table, new-seed and independent validation remain open. |
| B5 | Learned model selection of runtime operations | V2/V3 establish bounded dispatch integration; V4/V5 cover all four actual XOR inputs in development. Broader programs and independent confirmation remain open. |
| Line C | LC1–LC6 validation targets | Local LC1 development acceptance is reported; LC2–LC6 and independent confirmation remain open. See [targets](LINE_C_VALIDATION_TARGETS.md). |

The unified legacy audits are not a shortcut to full B4/B5 confirmation:
[v2.4.1](../evidence/unified_l1_l5_v2_4_1/THREE_SEED_SUMMARY.json) recorded 0/3
all-gates passes; [v2.4.3](../evidence/unified_l1_l5_v2_4_3/THREE_SEED_SUMMARY.json)
recorded 2/3 after an expression-coverage repair. The latter is not untouched
replication or a unified 3/3 confirmation.

## Evidence index and version history

| Archive | Interpretation |
| --- | --- |
| [B5 complete XOR V5](../evidence/gpt2_b5_xor_continuation_a100_v5/) | Latest four-input development coverage, 4/4 across inherited V4 records and V5 continuation; original gates. |
| [B5 full XOR V4](../evidence/gpt2_b5_xor_truth_table_a100_v4/) | Historical parent failure, 2/4; negative cases retained. |
| [B5 single-backbone V3](../evidence/gpt2_b5_single_backbone_a100_v3/) | Earlier partitioned-adapter development evidence; external routing and updates. |
| [B5 trained dispatch V2](../evidence/gpt2_b5_trained_dispatch_a100_v2/) | Two-model learned-dispatch development evidence; preserved launcher and result ZIP. |
| [B4 minimal causal V1](../evidence/gpt2_b4_minimal_causal_a100_v1/) | Central causal demonstration, fixed input `01`, development seed `82931`. |
| [Earlier B4 XOR V1](../evidence/gpt2_b4_causal_xor_a100_v1/) | Historical limitations retained; the substituted branch did not require an output flip and reset sensitivity was weak. |
| [Full XOR B4 V2](../external_tests/gpt2_b4_full_xor_a100_v2/) | Frozen protocol, results pending. |
| [L1 v1.3.2](../evidence/l1_exact_anchoring_v1_3_2/) | Exact anchoring development pass; secondary paraphrase gates failed. |
| [L2 v1.5.6](../evidence/l2_heldout_expression_v1_5_6/) | Source/protocol only; standalone result archive missing. |
| [L3 writer v1.7.8](../evidence/l3_writer_v1_7_8/) | Compositional writing development record. |
| [L3 category v1.8.1](../evidence/l3_category_v1_8_1/) | Partial development result. |
| [L3 v1.9.0 / v1.9.1](../evidence/l3_category_v1_9_1/) | Development result plus prospective follow-up protocol. |
| [L3 integration v1.9.6.0](../evidence/l3_category_v1_9_6_0/) | Single-seed integration record; legacy manifest check currently blocked as noted above. |
| [L4 v2.1.2](../evidence/l4_transport_v2_1_2/) | Local endpoint transport; dependent pair comparisons do not constitute independent replication. |
| [L5 v2.2.4 / B3](../evidence/l5_programmable_v2_2_4/) | Shared-state operations within the paper's declared scope. |
| [Joint v2.3.2.1](../evidence/joint_l1_l5_v2_3_2_1/) | Post-failure development smoke, not confirmation. |
| [Unified v2.4.1](../evidence/unified_l1_l5_v2_4_1/) | Prospective negative localization record. |
| [Unified v2.4.3](../evidence/unified_l1_l5_v2_4_3/) | Post-failure expression-coverage repair. |

The [legacy ladder](MFI_L1_L5_EVIDENCE_LADDER.md),
[previous README](README_HISTORY_2026-10-11.md),
[response-fibre foundations](RESPONSE_FIBRE_FOUNDATIONS.md) and
[older historical index](HISTORICAL_README_2026-09-05.md) retain version details
and links for the geometry, Pareto, Picard, GPTW, Pythia and ResNet branches.
These branches have their own claims; they are not additional B4/B5 replications.
