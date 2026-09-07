# Moving Fibre Intelligence

## Reusable geometric addressing in neural response fibres

Moving Fibre Intelligence tests whether memories and concepts can be assigned
to reusable geometric addresses inside a response-preserving parameter fibre.

For a prospectively declared finite response map \(R(\theta)\), the local
response-preserving space is

\[
V_\theta=\ker DR(\theta).
\]

The programme asks whether movement in this space can support exact memory,
semantic access, reusable concept addressing and eventually cross-fibre
transport while remaining within frozen response and KL budgets.

The current programme is organized as four experimentally gated layers:

**Exact memory -> semantic access -> concept geometry -> cross-fibre transport**

| Layer | Question | Current evidence status |
| --- | --- | --- |
| **L1 — exact memory** | Can a token-level memory be written to and read from a response-preserving fibre position? | **Single-seed mechanism supported** |
| **L2 — semantic access** | Can multiple linguistic expressions access the same written memory, including overwrite? | **Single-seed mechanism supported** |
| **L3 — compositional writer** | Can reusable two-slot concept codes be written, mixed and reassigned? | **v1.7.8 single-seed all gates passed** |
| **L3 — category geometry** | Can held-out concepts be routed by category-coordinate geometry under declared controls? | **v1.9.0 single-seed development all gates passed** |
| **L3 — independent confirmation** | Does the integrated router-address-writer mechanism survive a prospectively frozen fresh seed? | **v1.9.6.0 single-seed confirmation** |
| **L4 — transport** | Can concept structure move across fibres through a declared connection while preserving readable relations? | **v2.1.2 single-seed local transport candidate** |

These labels describe the evidence ladder, not four proven mathematical
theorems. Multi-seed confirmation and model-family replication remain required.

## Geometry resource status

This table states what each geometric term is currently allowed to mean in the
archive. It is a claim-boundary table, not a list of mathematical theorems.

| Resource | Observable | Status | Current interpretation |
| --- | --- | --- | --- |
| Position writing on response fibres | Write/read/overwrite accuracy | Single-seed support | Memories can be anchored to repeatably readable fibre positions within declared response budgets. |
| Addressable slot structure | Independent `amber`/`cedar` slot access | Single-seed support | Two declared slots can independently carry and retrieve written memory content. |
| Compositional coding | Two-bit codes, mixed-slot margins and cyclic reassignment | Single-seed support | The same slot-code rule can be composed and reused across written concepts. |
| Expression-level access | Held-out phrasings for written concepts | Single-seed support | Different expressions can access the same previously written memory; this does not prove that all semantic equivalents occupy identical internal coordinates. |
| Category-level generalization | Unseen whole-concept routing and positive margins | Single-seed confirmation | v1.9.6.0 routed 4/4 new bridge concepts correctly after a fresh-split 80-concept router passed 79/80 outer-held-out predictions. |
| Category geometry | Within/between distance ratio plus shuffled, pair-only and no-graph controls | Single-seed development support | v1.9.0 passed the declared distance-ratio and beat shuffled, pair-only and no-graph controls in one adaptive-development seed; v1.9.6.0 confirms the integrated router-address-writer mechanism, not L4 transport. |
| Geodesic semantic distance | Path distance predicts semantic relations | Untested | Current Euclidean or projected distances are not certified semantic geodesics. |
| Curvature | Local density, relation complexity or path deviation | Untested | No curvature estimator or curvature control has passed. |
| Parallel transport | Relation preservation after moving concept structures across base points | Local single-seed candidate | v2.1.2 transported an eight-address atlas between one source and one target finite-response level set and back; this is not global or path-independent parallel transport. |
| Fibre connection | Cross-fibre direction choice, path dependence and holonomy controls | Local protocol candidate | v2.1.2 supplies a tested local transport rule in one GPT-2 LoRA chart, not a certified global connection. |
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

### L3 category geometry — v1.9.0 / v1.9.6.0

In frozen seed `82001`, v1.9.0 passed the declared single-seed L3 development
gates:

- 8/8 held-out whole-concept category routes;
- minimum held-out distance margin `+1.9900`;
- within/between category-coordinate ratio `0.0301`;
- superiority to shuffled-category, pair-only and no-graph controls.

The development split influenced mechanism design, so v1.9.0 is single-seed
mechanism support rather than independent confirmation.

v1.9.6.0 is the first prospectively frozen single-seed confirmation of the
integrated L3 router-address-writer mechanism in this GPT-2 LoRA setting. It
used writer seed `82601`, router split seed `196001`, new bridge instances
`dog`, `Tokyo`, `Japan` and `apple`, and unchanged rank-8 writer budgets and
decision gates. The fresh-split router reached 79/80 outer-held-out accuracy
with 93.75% worst-fold accuracy and 95% minimum class recall; bridge routing
was 4/4 correct with positive margins; 16/16 eligible writer endpoints passed
exact, training-expression, held-out-expression, mixed-address, response-budget
and endpoint-KL gates.

[Inspect the frozen v1.9.6.0 category evidence](evidence/l3_category_v1_9_6_0/)

This is not multi-seed confirmation, a fresh external concept cohort,
cross-model replication or L4 cross-fibre transport. Earlier v1.8.1 partial
and v1.9.1 prospective records are preserved at
[`evidence/l3_category_v1_8_1/`](evidence/l3_category_v1_8_1/) and
[`evidence/l3_category_v1_9_1/`](evidence/l3_category_v1_9_1/).

### L4 local cross-fibre transport — v2.1.2

The prospective single-seed large-atlas audit transports an eight-address atlas
from a source finite-response level set to a distinct target level set and
back. At response shift `0.08`, transport preserved exact and held-out access,
reached the target response set, satisfied the collateral-KL budget, preserved
pair geometry and completed the round trip.

Mean relative geometry distortion was `1.45136e-05`, versus a matched-random
median of `7.88815e-05` (ratio `0.1840`). Transport won all 28 dependent
pair-distance comparisons. The 28 distances share eight endpoints, so this is
a descriptive win count; the archived sign-test value is retained only as a
naive dependent-pair diagnostic.

[Inspect the frozen v2.1.2 L4 transport evidence](evidence/l4_transport_v2_1_2/)

This is local evidence in one GPT-2 LoRA chart, not global parallel transport,
path-independent transport, multi-seed confirmation or cross-model transport.

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
evidence/l3_category_v1_9_1/
evidence/l3_category_v1_9_6_0/
evidence/l4_transport_v2_1_2/
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
