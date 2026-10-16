# Moving Fibre Intelligence

Research on response-constrained parameter geometry and programmable neural
memory. This repository contains separate experimental lines: GPT-2 parametric
memory, response-fibre geometry and optimization, and Qwen development/interface
experiments. Their acceptance criteria and evidence statuses are not interchangeable.

> **Start here: [Programmable Parametric Memory: From Selective Writes to Model-Selected Execution](docs/PROGRAMMABLE_PARAMETRIC_MEMORY.md).**
> A research framework for fixed-access parametric memory, with selective updates,
> declared response-preservation budgets, and causal read–compute–write tests.
>
> **Current development focus:** frozen single-proposal parametric writing, C35–C37.
> Both composite branches pass 16/16, 32/32 and 32/32 new local states, respectively.
> [Line C results and limits](docs/LINE_C_LOCAL_WRITING.md). Same-model/chart development; independent confirmation pending.
> **Operational foundation:** B4 causal read–compute–write and B5 V5 four-input XOR coverage; external runtime execution remains.
> [B5 V5 results and read-only verifier](evidence/gpt2_b5_xor_continuation_a100_v5/)
> · [V4 negative parent: 2/4](evidence/gpt2_b5_xor_truth_table_a100_v4/)
> · [B5 V3 archive](evidence/gpt2_b5_single_backbone_a100_v3/)
> · [B5 V2 archive and reproduction assets](evidence/gpt2_b5_trained_dispatch_a100_v2/)
> · [B4 causal evidence](evidence/gpt2_b4_minimal_causal_a100_v1/)

## Frozen local writing: C35–C37 development evidence

**Latest Line C result:** two frozen composite writers pass all declared gates
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

[Results, gates and provenance](docs/LINE_C_LOCAL_WRITING.md) ·
[LC1–LC6 validation targets](docs/LINE_C_VALIDATION_TARGETS.md)

## Research question and contribution

Can a restricted neural parameter chart expose fixed-access memory cells that
can be selectively updated, read and used in causal computations while declared
responses stay within finite budgets? The contribution is an operational memory
interface, explicit acceptance protocols, causal interventions and a traceable
chain of experiments. Standard null-space projection supplies a local mechanism;
it is not presented as a new mathematical discovery.

## Earlier archive-supported evidence and current status

| Research line | Current archive-supported status | Scope |
| --- | --- | --- |
| GPT-2 B0–B3 foundations | Reported results within the [paper's scope](paper/A_Programmable_Address_Space_for_Neural_Memory.pdf), archived under legacy L1–L5 labels | Separate selective-write, addressing and transport audits; no unified all-gates multi-seed confirmation. |
| **B4 minimal causal XOR** | **`B4_MINIMAL_CAUSAL_DEVELOPMENT_PASS`** — [raw summary](evidence/gpt2_b4_minimal_causal_a100_v1/recorded/SUMMARY.json) | One development seed, fixed input `01`; actual parameter reads, external XOR, parameter writes, wrong-read/input-substitution/reset/restore interventions. |
| **B5 V2 learned model dispatch** | **Two-model learned-dispatch development pass** — [raw summary](evidence/gpt2_b5_trained_dispatch_a100_v2/recorded/SUMMARY.json) | Separate controller and memory GPT-2 models; 8/8 symbolic readiness episodes, three positive physical cases and a reset control. |
| B4 full XOR V2 | [Protocol ready; real-model results pending](external_tests/gpt2_b4_full_xor_a100_v2/STATUS.json) | Must remain distinct from the completed minimal B4 experiment. |
| **B5 V3** | **Single-backbone, partitioned-adapter development pass** — [raw summary](evidence/gpt2_b5_single_backbone_a100_v3/recorded/SUMMARY.json) | One GPT-2 backbone, separate controller/memory adapters; external routing, XOR and numerical writes. |
| **B5 complete XOR V5** | **`B5_XOR_CONTINUATION_DEVELOPMENT_PASS`, 4/4** — [raw summary](evidence/gpt2_b5_xor_continuation_a100_v5/recorded/SUMMARY.json) | 00/01 inherited from V4; 10/11 pass after development continuation. All four require output flips; original endpoint gates remain fixed. |
| B5 full XOR V4 (historical parent) | [Development failure: 2/4](evidence/gpt2_b5_xor_truth_table_a100_v4/) | 10 preparation failed; 11 candidate writeback rejected. Original failures retained. |
| B5 independent confirmation | Pending | New controller seed does not establish independent memory-seed confirmation. |
| Qwen | [Separate evidence lines](docs/PROGRAMMABLE_PARAMETRIC_MEMORY.md#qwen-status-and-missing-version-evidence) | V8 incomplete, V9 frozen-interface audit, R23 optimizer experiments; V23/fresh-evaluation claims are not verified in this checkout. |
| Line C | C35–C37 local composite-writer passes are reported in execution logs | See the new results above; independent result-ZIP audit and update internalisation remain open. |

The latest [V5 archive](evidence/gpt2_b5_xor_continuation_a100_v5/) covers all four
actual XOR input preparations with initially incorrect outputs. It combines
inherited V4 passes for `00`/`01` with successful V5 continuation for `10`/`11`;
the latter retain true candidate and optimizer checkpoints. All final read
margins exceed `0.05`, averaged response drift is approximately `1.53e-5`
(budget `0.002`) and mean anchor KL is below `0.005`. The controller receives
no new training. This is development coverage, not independent confirmation
or Boolean completeness. XOR and numerical writes remain runtime services.

B4 supplies the earlier causal demonstration for this research line. B5 adds
learned selection of READ, XOR, WRITE and HALT in a bounded runtime. Its 8/8
readiness result is **two training-excluded address tuples × four symbolic XOR
inputs**, not eight actual parameter-memory experiments. The physical cases are
`normal`, `stored_substitution`, `restored` and the `reset` control. The
[topic page explains why reset retains `pass=false`](docs/PROGRAMMABLE_PARAMETRIC_MEMORY.md#b5-v2-two-model-development).

V3's controller seed `82933` reuses memory seed `82931`, B4 states and task
combinations already inspected in V2. Three positive physical cases pass; reset
retains `pass=false` solely for no output flip. The fixed controller score probe
changes by `0` across four recorded memory writes, consistent with adapter
isolation. This covers only the declared probe and does not demonstrate
self-modification in one shared update region. See the
[V3 architecture and evidence boundaries](docs/PROGRAMMABLE_PARAMETRIC_MEMORY.md#b5-v3-single-backbone-development).

These records do not establish complete independent B5 confirmation, arbitrary
program execution, single-model native weight updates, lifelong learning or zero
forgetting. XOR and numerical parameter updates remain external runtime services.

## Verify without a GPU

From the repository root, with Python 3 and no model download:

```bash
python3 evidence/gpt2_line_c_c35_c37/verify_reported_evidence.py
python3 evidence/gpt2_b5_xor_continuation_a100_v5/verify_archive.py
python3 evidence/gpt2_b5_xor_truth_table_a100_v4/verify_archive.py
python3 evidence/gpt2_b5_single_backbone_a100_v3/verify_archive.py
python3 evidence/gpt2_b5_trained_dispatch_a100_v2/verify_archive.py
python3 evidence/gpt2_b4_minimal_causal_a100_v1/verify_archive.py
```

These read-only checks verify archive hashes and recorded consistency. They do
not re-execute GPT-2 or independently reproduce model readouts. See the
[verification guide and known legacy blocker](docs/PROGRAMMABLE_PARAMETRIC_MEMORY.md#verify-archives-without-a-gpu).

## Run an actual A100 experiment

For the latest four-input development continuation, use the
[B5 V5 one-cell PY](evidence/gpt2_b5_xor_continuation_a100_v5/GPT2_B5_XOR_CONTINUATION_A100_V5.py)
with the [verified failed V4 result](evidence/gpt2_b5_xor_truth_table_a100_v4/gpt2_b5_xor_truth_table_a100_results.zip).
A matching [V5 result ZIP](evidence/gpt2_b5_xor_continuation_a100_v5/gpt2_b5_xor_continuation_a100_results.zip)
restores V5 progress; the published completed archive is not a new experiment.
The wrapper includes a dependency-directory repair while preserving the archived
worker and resume identity. [V5 scope and input guide](evidence/gpt2_b5_xor_continuation_a100_v5/).

Open the [B5 V2 one-cell Colab launcher](evidence/gpt2_b5_trained_dispatch_a100_v2/GPT2_B5_TRAINED_DISPATCH_A100_V2.py)
and copy the entire file into a Colab notebook with an A100 GPU. This performs
actual model execution and training. Supply the passed
[B4 minimal-causal result ZIP](evidence/gpt2_b4_minimal_causal_a100_v1/gpt2_b4_minimal_causal_a100_results.zip)
for a new B5 V2 run, or a matching
[B5 V2 result ZIP](evidence/gpt2_b5_trained_dispatch_a100_v2/gpt2_b5_trained_dispatch_a100_results.zip)
for resume. **Upload a result ZIP, not a source or launcher ZIP.**

For the single-backbone experiment, use the [B5 V3 one-cell launcher](evidence/gpt2_b5_single_backbone_a100_v3/GPT2_B5_SINGLE_BACKBONE_A100_V3.py).
It accepts the same B4 result, or extracts the B4 parent from a V2 result, to
train a new V3 controller; only a matching [V3 result ZIP](evidence/gpt2_b5_single_backbone_a100_v3/gpt2_b5_single_backbone_a100_results.zip)
resumes V3. Uploading completed results for resume is not a fresh replication.

[Detailed input protocols, Drive resume behavior and B4 launchers](docs/PROGRAMMABLE_PARAMETRIC_MEMORY.md#a100-experiments-and-input-files).

## Mechanism, roadmap and evidence

For the declared protected response map, its Jacobian is \(J\); \(Jv=0\)
expresses first-order preservation. If \(N\) spans the permitted response kernel
and \(H\) is the readout map, \(C=DH\,N\) describes local readout controllability
within those directions. Finite updates remain limited by curvature, numerical
approximation, response/KL budgets and parameter constraints.

- [Mathematical scope and B0–B5 / Line C roadmap](docs/PROGRAMMABLE_PARAMETRIC_MEMORY.md#mathematical-mechanism).
- [Parametric-memory evidence index](docs/PROGRAMMABLE_PARAMETRIC_MEMORY.md#evidence-index-and-version-history), with original summaries, protocols and historical limitations.
- [Response-fibre geometry, Pareto and optimization foundations](docs/RESPONSE_FIBRE_FOUNDATIONS.md) and [claim boundaries](docs/CLAIM_BOUNDARIES.md).
- [Qwen branch](fibre-qwen/README.md), separate from GPT-2 memory claims.
- [Previous README and full historical links](docs/README_HISTORY_2026-10-11.md).

## Papers and attribution

[A Programmable Address Space for Neural Memory](paper/A_Programmable_Address_Space_for_Neural_Memory.pdf)
reports the legacy L1–L5 stage; the B4–B5 archives are later repository evidence.
[Moving Response Fibres](paper/Moving_Response_Fibres_A_Geometric_View_of_Behaviour_and_Learning.pdf)
covers the geometry line. Cite the specific protocol and repository commit.
Author: **Y. Y. N. Li**. Repository and upstream model/data licenses apply independently.
