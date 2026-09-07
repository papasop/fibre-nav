# Moving Fibre Intelligence

## From addressable memory to programmable operations in neural parameter space

Moving Fibre Intelligence (MFI) asks whether a neural network can support an
explicit memory-address structure inside a constrained parameter chart. For a
prospectively declared finite response map \(R(\theta)\), the local
response-preserving directions are

\[
V_\theta=\ker DR(\theta).
\]

The experiments test whether movement in this restricted space can support
memory addresses that are exact, accessible from held-out expressions,
compositionally reusable, locally transportable and operable in a shared
parameter state—all while satisfying declared response and KL budgets.

> **Evidence arc:** L1–L4 construct and validate the address geometry. L5 is
> the culmination: WRITE, selective OVERWRITE, SWAP, MOVE and RETURN are
> audited together on one shared four-cell state.

## The L1–L5 evidence chain

| Layer | Capability established | Role in the argument | Archived status |
| --- | --- | --- | --- |
| **L1 — exact anchoring** | Write, read and overwrite a token-level memory at a controlled fibre endpoint. | Establishes that memory can be associated with position. | v1.3.2 single-seed development evidence; exact gates passed, paraphrase gates did not. |
| **L2 — held-out expression access** | Retrieve a written memory through expressions excluded from within-run writing updates. | Separates access from exact prompt repetition. | Standalone v1.5.6 protocol archived; public positive gates are contained in later L3–L5 audits. |
| **L3 — reusable compositional addressing** | Compose two independently readable binary slots into four codes and cyclically reassign them across concepts; route held-out concepts to addresses. | Separates concept identity from a reusable address code. | v1.7.8 development pass; v1.9.6.0 prospective single-seed integration candidate. |
| **L4 — local transport** | Move eight concept-specific endpoints assigned across four repeated two-slot codes between distinct finite-response level sets and back while preserving access and pairwise geometry. | Tests whether an instantiated address geometry survives a change of response constraint. | v2.1.2 prospective single-seed confirmation candidate. |
| **L5 — shared-state operations** | Jointly WRITE four cells, selectively OVERWRITE one, SWAP two, MOVE the resulting state and RETURN it. | Combines the preceding capabilities into an externally operated shared memory state. | **v2.2.4 prospective fresh-seed and fresh-expression single-seed confirmation candidate; all 16 declared gates passed.** |

This is a **layered evidence chain**, not one unified L1–L5 run and not a
sequence of mathematical theorems. Passing L5 does not retroactively replace
the archived L1–L4 protocols.

## Headline result: the L5 shared state

In seed `82931`, one restricted GPT-2 LoRA chart supported four jointly written
binary address cells. The audit then:

1. selectively overwrote address `01` while preserving unselected cells;
2. swapped the contents of addresses `00` and `11`;
3. moved the final shared state to a distinct finite-response level set; and
4. returned it to the source set.

Every program stage preserved exact and fresh held-out-expression access,
positive held-out margins, declared response and KL budgets, and all required
unselected addresses. OVERWRITE and SWAP beat wrong-operation and matched-norm
random-kernel controls. MOVE and RETURN preserved all four addresses.

The operation names denote **externally specified experimental procedures**.
WRITE, OVERWRITE and SWAP use response-constrained projected-gradient updates
to restricted LoRA coordinates; MOVE and RETURN use local response retraction.
GPT-2 does not parse or autonomously execute an instruction language, and the
result is not gradient-free inference-time memory.

## Supporting results

- **L1:** v1.3.2 passed the primary exact anchoring, overwrite and switching
  gates in one GPT-2 LoRA development seed.
- **L2 access:** held-out-expression retrieval is directly audited inside the
  archived L3 writer, L4 transport and L5 shared-state results. The standalone
  v1.5.6 result archive is unavailable, so the repository does not present it
  as an independent confirmation.
- **L3 writer:** v1.7.8 passed exact, train-expression, held-out-expression,
  mixed-slot, cyclic-reassignment, response and endpoint-KL gates in seed
  `81902`.
- **L3 integration:** v1.9.6.0 recorded 79/80 outer-held-out router predictions,
  4/4 bridge routes and 16/16 eligible writer endpoints using writer seed
  `82601` and router split seed `196001`.
- **L4:** v2.1.2 transported eight concept-specific endpoints assigned across
  four repeated two-slot codes, spanning 28 pairwise relations. Mean relative
  distortion was `1.45136e-05`, versus a random-kernel-perturbed retraction
  control median of `7.88815e-05`; transport won all 28 dependent pairwise
  comparisons and preserved round-trip access.

## What the evidence does—and does not—support

Within the tested GPT-2 LoRA charts, the archive supports a finite,
compositional and remappable address structure; held-out-expression access to
written values; local endpoint-wise transport; and externally executed shared
state operations.

It does **not** establish:

- autonomous instruction parsing or a general neural instruction-set
  architecture;
- gradient-free or ordinary inference-time memory operations;
- arbitrary program composition or algebraic closure over arbitrary states;
- semantic geodesics, curvature, a certified fibre connection or a global
  section;
- multi-seed MFI confirmation, cross-model replication, global/path-independent
  parallel transport or concurrent storage beyond the declared four cells.

Here, *response preservation* concerns the declared finite response map, not
the model's complete input–output function. A passed single seed is local
mechanism evidence, not model-family confirmation.

See [Claim Boundaries](docs/CLAIM_BOUNDARIES.md) and the
[MFI L1–L5 Evidence Ladder](docs/MFI_L1_L5_EVIDENCE_LADDER.md).

## Reproduce and verify

Run the repository-level archive checks:

```bash
python verify_mfi_l1_l5.py
python verify_evidence.py
```

`verify_mfi_l1_l5.py` checks the archived MFI ladder. The older
`verify_evidence.py` covers the separate response-fibre and Pareto branches;
its `PC2 remains blocked` message records an expected boundary in that branch.
Each evidence directory also contains its own source, results, provenance,
claim boundary and reproduction instructions where available.

## Evidence index

- [L1 exact anchoring — v1.3.2](evidence/l1_exact_anchoring_v1_3_2/)
- [L2 held-out-expression protocol — v1.5.6](evidence/l2_heldout_expression_v1_5_6/)
- [L3 compositional writer — v1.7.8](evidence/l3_writer_v1_7_8/)
- [L3 category development archive — v1.9.1](evidence/l3_category_v1_9_1/)
- [L3 integrated confirmation candidate — v1.9.6.0](evidence/l3_category_v1_9_6_0/)
- [L4 local transport — v2.1.2](evidence/l4_transport_v2_1_2/)
- [L5 programmable operations — v2.2.4](evidence/l5_programmable_v2_2_4/)

## Repository map

```text
evidence/                 frozen, developmental and audit records
external_tests/           one-click external reproductions
docs/                     claim boundaries, evidence ladders and history
paper/                    manuscript snapshots
provenance/               hashes and provenance records
fibre-qwen/               separate developmental assistant overlay
```

Earlier response-fibre, Pareto, Picard, GPTW, Pythia, ResNet and Fibre-Qwen
branches are retained as foundations and historical evidence, but they are not
part of the current MFI L1–L5 headline claim. See
[Response Fibre Foundations](docs/RESPONSE_FIBRE_FOUNDATIONS.md) and the
[Historical README](docs/HISTORICAL_README_2026-09-05.md).

## Paper, citation and license

The current L1–L5 manuscript is not yet archived in this repository. The
historical manuscript snapshot is
[Moving Response Fibres: A Geometric View of Behaviour and Learning](paper/Moving_Response_Fibres_A_Geometric_View_of_Behaviour_and_Learning.pdf).

Author: **Y. Y. N. Li**. Cite the exact repository commit and evidence protocol
used. The repository license and the licenses of upstream models and datasets
apply independently.
