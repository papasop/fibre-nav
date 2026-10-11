# Moving Fibre Intelligence

Research on response-constrained parameter geometry and programmable neural
memory. This repository contains separate experimental lines: GPT-2 parametric
memory, response-fibre geometry and optimization, and Qwen development/interface
experiments. Their acceptance criteria and evidence statuses are not interchangeable.

> **Start here: [Programmable Parametric Memory: From Selective Writes to Model-Selected Execution](docs/PROGRAMMABLE_PARAMETRIC_MEMORY.md).**
> A research framework for fixed-access parametric memory, with selective updates,
> declared response-preservation budgets, and causal read–compute–write tests.
>
> **Empirical center:** B4 minimal causal read–compute–write.
> **Latest development:** B5 V3 single-backbone, partitioned-adapter dispatch.
> [B5 V3 archive](evidence/gpt2_b5_single_backbone_a100_v3/)
> · [B5 V2 archive and reproduction assets](evidence/gpt2_b5_trained_dispatch_a100_v2/)
> · [B4 causal evidence](evidence/gpt2_b4_minimal_causal_a100_v1/)

## Research question and contribution

Can a restricted neural parameter chart expose fixed-access memory cells that
can be selectively updated, read and used in causal computations while declared
responses stay within finite budgets? The contribution is an operational memory
interface, explicit acceptance protocols, causal interventions and a traceable
chain of experiments. Standard null-space projection supplies a local mechanism;
it is not presented as a new mathematical discovery.

## Strongest evidence and current status

| Research line | Current archive-supported status | Scope |
| --- | --- | --- |
| GPT-2 B0–B3 foundations | Reported results within the [paper's scope](paper/A_Programmable_Address_Space_for_Neural_Memory.pdf), archived under legacy L1–L5 labels | Separate selective-write, addressing and transport audits; no unified all-gates multi-seed confirmation. |
| **B4 minimal causal XOR** | **`B4_MINIMAL_CAUSAL_DEVELOPMENT_PASS`** — [raw summary](evidence/gpt2_b4_minimal_causal_a100_v1/recorded/SUMMARY.json) | One development seed, fixed input `01`; actual parameter reads, external XOR, parameter writes, wrong-read/input-substitution/reset/restore interventions. |
| **B5 V2 learned model dispatch** | **Two-model learned-dispatch development pass** — [raw summary](evidence/gpt2_b5_trained_dispatch_a100_v2/recorded/SUMMARY.json) | Separate controller and memory GPT-2 models; 8/8 symbolic readiness episodes, three positive physical cases and a reset control. |
| B4 full XOR V2 | [Protocol ready; real-model results pending](external_tests/gpt2_b4_full_xor_a100_v2/STATUS.json) | Must remain distinct from the completed minimal B4 experiment. |
| **B5 V3** | **Single-backbone, partitioned-adapter development pass** — [raw summary](evidence/gpt2_b5_single_backbone_a100_v3/recorded/SUMMARY.json) | One GPT-2 backbone, separate controller/memory adapters; external routing, XOR and numerical writes. |
| B5 independent confirmation | Pending | New controller seed does not establish independent memory-seed confirmation. |
| Qwen | [Separate evidence lines](docs/PROGRAMMABLE_PARAMETRIC_MEMORY.md#qwen-status-and-missing-version-evidence) | V8 incomplete, V9 frozen-interface audit, R23 optimizer experiments; V23/fresh-evaluation claims are not verified in this checkout. |
| Line C | Open research objective | Internalizing parameter updates is not established by B5's external numerical writer. |

B4 is the causal demonstration at the center of this research line. B5 adds
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
python3 evidence/gpt2_b5_single_backbone_a100_v3/verify_archive.py
python3 evidence/gpt2_b5_trained_dispatch_a100_v2/verify_archive.py
python3 evidence/gpt2_b4_minimal_causal_a100_v1/verify_archive.py
```

These read-only checks verify archive hashes and recorded consistency. They do
not re-execute GPT-2 or independently reproduce model readouts. See the
[verification guide and known legacy blocker](docs/PROGRAMMABLE_PARAMETRIC_MEMORY.md#verify-archives-without-a-gpu).

## Run an actual A100 experiment

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
