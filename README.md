# Moving Fibre Intelligence

## Reusable geometric addressing in neural response fibres

Moving Fibre Intelligence tests whether memories and concepts can be assigned
to reusable geometric addresses inside a response-preserving parameter fibre.
For a prospectively declared finite response map \(R(\theta)\), the local
response-preserving space is

\[
V_\theta=\ker DR(\theta).
\]

We test whether movement in this space can support exact memory, semantic
access, reusable concept addressing, local transport and externally specified
address operations while remaining within frozen response and KL budgets.

**Exact memory -> semantic access -> concept geometry -> cross-fibre transport -> programmable address operations**

| Layer | Question | Current evidence status |
| --- | --- | --- |
| **L1 — exact memory** | Can a token-level memory be written to and read from a response-preserving fibre position? | **Single-seed mechanism supported** |
| **L2 — expression access** | Can held-out expressions read a written memory? | **Standalone result unavailable; public evidence comes from L3/L4 gates** |
| **L3 — compositional writer** | Can reusable two-slot concept codes be written, mixed and reassigned? | **v1.7.8 single-seed all gates passed** |
| **L3 — category geometry** | Can held-out concepts be routed by category-coordinate geometry under declared controls? | **v1.9.0 development pass; v1.9.6.0 single-seed confirmation candidate** |
| **L4 — local transport** | Can concept endpoints move between finite-response level sets by a declared local transport rule while preserving readable relations? | **v2.1.2 single-seed local transport candidate** |
| **L5 — programmable operations** | Can externally specified WRITE, OVERWRITE, SWAP, MOVE and RETURN procedures operate on a shared address state? | **v2.2.4 single-seed confirmation candidate** |

These labels describe the MFI L1-L5 evidence ladder, not proven mathematical
theorems. Multi-seed confirmation and model-family replication remain open.

## Quick verification

```bash
python verify_mfi_l1_l5.py
python verify_evidence.py
```

`verify_mfi_l1_l5.py` checks the MFI archive status directly. The older
`verify_evidence.py` checks the response-fibre and Pareto branches; its
`PC2 remains blocked` message is an expected boundary for that separate branch.

## Key results

| Result | Evidence |
| --- | --- |
| L1 exact anchoring | v1.3.2 archives one GPT-2 LoRA development seed with primary exact anchoring and overwrite gates passed; secondary paraphrase gates did not pass. |
| L2 expression access | Standalone v1.5.6 source and protocol are archived, but the original development result is unavailable. Held-out-expression access is instead supported by archived L3 and L4 gates. |
| L3 writer | v1.7.8 passed all exact, training-expression, held-out-expression, mixed-slot, cyclic reassignment, response and endpoint-KL gates in seed `81902`. |
| L3 category integration | v1.9.6.0 used writer seed `82601` and router split seed `196001`; 79/80 outer-held-out router predictions, 4/4 bridge routes and 16/16 eligible writer endpoints passed. |
| L4 local endpoint transport | v2.1.2 moved eight concept-specific endpoints assigned across four repeated two-slot codes from one source response set to one target response set and back. |
| L5 programmable address operations | v2.2.4 passed all 16 WRITE, selective OVERWRITE, SWAP, MOVE and RETURN gates in fresh seed `82931`. |

## Geometry resource status

| Validated resource | Observable | Interpretation |
| --- | --- | --- |
| Position writing | Write/read/overwrite accuracy | L1 supports exact anchoring within one development seed, not a general memory theorem. |
| Addressable slot structure | Independent `amber`/`cedar` slot access | L2 standalone results are unavailable; slot access is public through later archived L3/L4 records. |
| Compositional coding | Two-bit codes, mixed-slot margins and cyclic reassignment | The same slot-code rule can be composed and reused across written concepts in the tested chart. |
| Expression-level access | Held-out phrasings for written concepts | Different expressions can access the same previously written memory; this does not prove that all semantic equivalents occupy identical internal coordinates. |
| Category-level routing | Held-out whole-concept routes and positive margins | L3 category support is single-seed and chart-local. |
| Local endpoint-wise response transport | Access and pair-geometry preservation after endpoint movement | L4 is local retraction transport, not global or path-independent parallel transport. |
| Programmable address operations | WRITE, OVERWRITE, SWAP, MOVE and RETURN on one shared four-cell state | L5 operations are externally specified experimental procedures, not autonomous GPT-2 instruction execution. |

| Open geometric resource | Status |
| --- | --- |
| Geodesic semantic distance | Untested; current distances are not certified semantic geodesics. |
| Curvature | Untested; no curvature estimator or curvature control has passed. |
| Fibre connection | Untested; v2.1.2 supplies a local endpoint-wise retraction audit, not an experimentally certified connection. |
| Section | Untested; no local or global section has been established. |
| Multi-seed and model-family MFI confirmation | Open for L3/L4/L5. |

## Evidence map

- [L1 exact anchoring v1.3.2](evidence/l1_exact_anchoring_v1_3_2/)
- [L2 held-out-expression protocol v1.5.6](evidence/l2_heldout_expression_v1_5_6/)
- [L3 compositional writer v1.7.8](evidence/l3_writer_v1_7_8/)
- [L3 category development v1.9.1 archive](evidence/l3_category_v1_9_1/)
- [L3 integrated category confirmation candidate v1.9.6.0](evidence/l3_category_v1_9_6_0/)
- [L4 local transport v2.1.2](evidence/l4_transport_v2_1_2/)
- [L5 programmable address operations v2.2.4](evidence/l5_programmable_v2_2_4/)

## Upstream response-fibre foundations

The repository also preserves earlier response-fibre, Pareto, Picard, GPTW,
Pythia, ResNet and Fibre-Qwen branches. They motivate the finite response-kernel
construction but are not part of the current MFI L1-L5 headline claim.

See [Response Fibre Foundations](docs/RESPONSE_FIBRE_FOUNDATIONS.md) for the
older evidence ladder and [Historical README](docs/HISTORICAL_README_2026-09-05.md)
for the pre-MFI repository narrative.

## Claim discipline

In this repository:

- *behaviour* means the value of the declared finite response map, not the
  model's complete input-output function;
- *learning* means improvement on the declared objective and split;
- a passed single seed is local mechanism evidence, not model-family
  confirmation;
- failed, partial and result-pending stages remain visible;
- L3 does not imply geodesic distance or curvature;
- L4 requires a transport rule, source and target response sets, matched
  controls and post-transport gates; certification of a fibre connection is a
  separate, untested step;
- L5 does not establish autonomous instruction following, gradient-free
  inference-time memory, arbitrary program composition or a general
  instruction-set architecture.

See [Claim Boundaries](docs/CLAIM_BOUNDARIES.md) and
[MFI L1-L5 Evidence Ladder](docs/MFI_L1_L5_EVIDENCE_LADDER.md).

## Repository map

```text
evidence/                 frozen, developmental and audit records
external_tests/           one-click external reproductions
docs/                     MFI boundaries, ladders and historical notes
paper/                    manuscript snapshots
provenance/               hashes and provenance records
fibre-qwen/               separate developmental assistant overlay
```

## Papers

The current MFI L1-L5 manuscript is not yet archived in this repository.
The historical response-fibre manuscript snapshot is
[Moving Response Fibres: A Geometric View of Behaviour and Learning](paper/Moving_Response_Fibres_A_Geometric_View_of_Behaviour_and_Learning.pdf).

## Citation and license

Author: **Y. Y. N. Li**. Cite the exact repository commit and evidence protocol
used. The repository license and the licenses of upstream models and datasets
apply independently.
