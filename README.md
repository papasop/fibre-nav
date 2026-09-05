# Moving Fibre Intelligence

## Response-preserving memory, concept geometry and transport

`fibre-nav` is an evidence-first research repository for a geometric view of
behaviour and learning.

Moving Fibre Intelligence studies whether memories and concept structures can
be written, read, and transported through parameter-space directions that
preserve a declared external response. A prospectively declared finite response
map \(R(\theta)\) defines the local response-preserving space

\[
V_\theta=\ker DR(\theta).
\]

The central question is whether movement inside this space can carry useful
memory and learning structure while keeping the declared external response
within frozen finite budgets.

The current programme is organized as four experimentally gated layers:

**Exact memory -> semantic access -> concept geometry -> cross-fibre transport**

| Layer | Question | Current evidence status |
| --- | --- | --- |
| **L1 — exact memory** | Can a token-level memory be written to and read from a response-preserving fibre position? | **Single-seed mechanism supported** |
| **L2 — semantic access** | Can multiple linguistic expressions access the same written memory, including overwrite? | **Single-seed mechanism supported** |
| **L3 — concept geometry** | Can concept identity and composition be represented geometrically and generalized to unseen expressions or concepts? | **Writer supported; category geometry remains developmental** |
| **L4 — transport** | Can concept structure move across fibres through a declared connection while preserving readable relations? | **Not yet established** |

These labels describe the evidence ladder, not four proven mathematical
theorems. Multi-seed confirmation and model-family replication remain required.

## Geometry Resource Status

This table states what each geometric term is currently allowed to mean in the
archive. It is a claim-boundary table, not a list of mathematical theorems.

| Resource | Observable | Status | Current interpretation |
| --- | --- | --- | --- |
| Position writing on response fibres | Write/read/overwrite accuracy | Single-seed support | Memories can be anchored to repeatably readable fibre positions within declared response budgets. |
| Addressable slot structure | Independent `amber`/`cedar` slot access | Single-seed support | Two declared slots can independently carry and retrieve written memory content. |
| Compositional coding | Two-bit codes, mixed-slot margins and cyclic reassignment | Single-seed support | The same slot-code rule can be composed and reused across written concepts. |
| Expression-level access | Held-out phrasings for written concepts | Single-seed support | Different expressions can access the same previously written memory; this does not prove that all semantic equivalents occupy identical internal coordinates. |
| Category-level generalization | Unseen whole-concept routing and positive margins | Partial development | v1.8.1 reached 7/8 unseen concepts, but the full category gate did not pass. |
| Category geometry | Within/between distance ratio plus shuffled, pair-only and no-graph controls | Not established | Shuffled and no-graph controls were beaten, but distance-ratio and pair-only gates did not jointly pass. |
| Geodesic semantic distance | Path distance predicts semantic relations | Untested | Current Euclidean or projected distances are not certified semantic geodesics. |
| Curvature | Local density, relation complexity or path deviation | Untested | No curvature estimator or curvature control has passed. |
| Parallel transport | Relation preservation after moving concept structures across base points | L4 pending | No frozen parallel-transport protocol has passed. |
| Fibre connection | Cross-fibre direction choice, path dependence and holonomy controls | L4 pending | No experimentally certified connection is available. |
| Section | Consistent concept-position choices across base points | L4+ pending | No local or global section has been tested. |

## Current frontier

### L3 compositional writer — v1.7.8

In frozen seed `81902`, v1.7.8 passed all declared exact, training-expression,
held-out-expression, response-budget, endpoint-KL, mixed-slot and per-view
gates across all four two-bit codes and four cyclic code assignments.

This supports a compositional concept-position writer for previously written
concepts in one GPT-2 development seed. It does not establish category-level
generalization, geodesic semantic distance, curvature or fibre transport.

[Inspect the frozen v1.7.8 writer evidence](evidence/l3_writer_v1_7_8/)

### L3 shared-category writer — v1.8.1

In frozen seed `82001`, v1.8.1 reached:

- 16/16 training-concept routes;
- 7/8 whole-concept held-out routes;
- 24/24 eligible category moves;
- superiority to shuffled-category and no-graph controls.

It did **not** pass the complete L3 category gate. Positive held-out margin,
within/between distance ratio, pair-only advantage and universal held-out
writer access remained unsupported. The correct status is a partial
development result, not confirmed category geometry.

[Inspect the frozen v1.8.1 category evidence](evidence/l3_category_v1_8_1/)

v1.8.2 and later category-repair runs remain outside the repository until they
are separately frozen, audited and assigned an explicit claim boundary.

## What is already established outside the L1–L4 memory ladder

The repository also contains earlier evidence for moving response geometry and
response-constrained learning. These branches motivate the fibre construction
but must not be conflated with concept-memory confirmation.

| Evidence branch | Narrow supported statement | Location |
| --- | --- | --- |
| GPT-2 LoRA Picard r4/r5 | Task-specific time-to-equal-loss advantage over AdamW under the frozen protocol | [`evidence/audits/picard_gpt2_lora_r5_ten_step`](evidence/audits/picard_gpt2_lora_r5_ten_step/) |
| GPT-2 strict response Pareto R2 | Selected current-kernel directions retain held-out utility under finite response budgets | [`evidence/gpt2_lora_r2_strict`](evidence/gpt2_lora_r2_strict/) |
| GPT-2 dynamic-kernel R12/R13 | Current-response-kernel updates outperform declared matched-budget controls in the frozen pretrained GPT-2 audit | [`evidence/pretrained_gpt2_dynamic_kernel_r12_r13`](evidence/pretrained_gpt2_dynamic_kernel_r12_r13/) |
| Pythia-160M R17d | Metric-constrained Onsager update supported within one frozen LoRA chart and task pair | [`evidence/confirmed/pythia160m_metric_onsager_r17d`](evidence/confirmed/pythia160m_metric_onsager_r17d/) |
| CNN/ResNet moving fibres | Finite-chart moving-versus-fixed geometry, selected tangent value and transverse amplification | [`evidence/confirmed`](evidence/confirmed/) |
| GPTW v1.1.x | Restricted language-model correction-cost and adaptive-value audits | [`evidence/gptw_response_fibre_v1_1_0`](evidence/gptw_response_fibre_v1_1_0/) |

The detailed pre-L1–L4 narrative remains available in
[`docs/HISTORICAL_README_2026-09-05.md`](docs/HISTORICAL_README_2026-09-05.md).

## Claim discipline

In this repository:

- *behaviour* means the value of the declared finite response map, not the
  model's complete input-output function;
- *learning* means improvement on the declared objective and split;
- a passed single seed is mechanism evidence, not confirmation;
- failed and partial stages remain visible;
- first-order kernel membership alone is not sufficient evidence;
- L3 does not imply geodesic distance or curvature;
- L4 requires an explicit transport protocol, connection, controls and
  post-transport readout gates.

See [`docs/CLAIM_BOUNDARIES.md`](docs/CLAIM_BOUNDARIES.md) and
[`docs/EVIDENCE_LADDER.md`](docs/EVIDENCE_LADDER.md).

## Repository map

```text
evidence/confirmed/       frozen confirmatory evidence
evidence/developmental/   developmental and failed stages
evidence/l3_writer_v1_7_8/
evidence/l3_category_v1_8_1/
external_tests/           one-click external reproductions
docs/                     claim boundaries and evidence ladder
paper/                    current paper and archived revision
provenance/               hashes and provenance records
fibre-qwen/               separate developmental assistant overlay
```

Historical files are retained for auditability. Their presence does not make
them part of the current L1–L4 headline claim.

## Verify archived evidence

```bash
python verify_evidence.py
python verify_gpt2_lora_pareto_r2.py
python verify_gpt2_lora_global_budget_r10_r11.py
python verify_picard_finetune_v0_2_6.py
python verify_picard_gpt2_lora_v1_6.py
python verify_pythia_r17d.py
```

Individual frozen snapshots contain their own reproduction instructions and
SHA-256 manifests.

## Paper

The associated manuscript is archived at
[`paper/Moving_Response_Fibres_A_Geometric_View_of_Behaviour_and_Learning.pdf`](paper/Moving_Response_Fibres_A_Geometric_View_of_Behaviour_and_Learning.pdf).

## Citation and license

Author: **Y. Y. N. Li**. Cite the exact repository commit and evidence protocol
used. The repository license and the licenses of upstream models and datasets
apply independently.
