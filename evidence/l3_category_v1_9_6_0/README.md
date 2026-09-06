# Moving Fibre Intelligence: Reusable Concept Addressing

This archive records the first prospectively frozen single-seed confirmation of the MFI L3 router–address–writer mechanism in the tested GPT-2 LoRA setting.

## Research question

Can a continuous response-preserving parameter set support a discrete, compositional, and remappable address structure whose codes can be reassigned across concept identities?

## L1–L3 experimental ladder

| Level | Criterion | Tested capability |
|---|---|---|
| L1 | Exact anchoring | Controlled memory writing, reading, and overwrite |
| L2 | Held-out expression access | Unseen formulations retrieve the same written memory |
| L3 | Reusable compositional addressing | Two slots and four binary codes remain readable under cyclic reassignment |

The integrated pipeline is:

```text
held-out description → semantic category → fixed two-bit address → rank-8 fibre writer → audited readout
```

## Frozen confirmation

The confirmation was frozen before execution with:

- writer and router seed: `82601`;
- stratified split seed: `196001`;
- new bridge instances: dog, Tokyo, Japan, and apple;
- unchanged rank-8 writer, optimizer, step count, budgets, and decision gates;
- held-out expressions excluded from optimization and checkpoint selection.

### Results

| Measure | Result |
|---|---:|
| Router outer-held-out accuracy | 79/80 (98.75%) |
| Worst-fold accuracy | 93.75% |
| Minimum class recall | 95% |
| Bridge routing | 4/4 correct, all positive margins |
| Writer endpoints | 16/16 eligible |
| Exact, training-expression, and held-out-expression access | 100% across all audited runs |
| Cyclic assignments | 4/4 passed |
| Complete declared gate family | **PASS** |

The minimum mixed-address training margin was `1.6406` against a gate of `0.5`; the minimum held-out margin was `1.5923`. Maximum response drift was approximately `2.29e-5` against a budget of `0.002`, and maximum endpoint KL was approximately `0.004095` against a gate of `0.005`.

## Scientific status

This archive supports a **prospectively frozen single-seed confirmation within the declared GPT-2 LoRA setting**. It is not multi-seed confirmation, a fresh-cohort external replication, cross-model replication, or cross-fibre transport. The bridge instances are new to this frozen run but are drawn from the existing 80-concept cohort.

Structure-preserving transport between distinct response fibres (L4) is not tested here.

## Repository layout

- `src/moving_fibre_intelligence_l3_v1_9_6_0/` — frozen rank-8 experiment
- `src/moving_fibre_intelligence_l3_router_v1_9_6_0/` — fresh-split semantic router
- `src/moving_fibre_intelligence_l3_bridge_v1_9_6_0/` — router/writer bridge
- `src/moving_fibre_intelligence_l3_v1_7_8/` — frozen writer dependency retained for exact reproduction
- `colab/` — notebook-safe launcher
- `release_assets/` — upload-ready launcher and source archive
- `evidence/v1_9_6_0/` — complete returned result archive and compact records
- `REPRODUCE.md` — execution instructions
- `CLAIM_BOUNDARY.md` — supported and unsupported claims

## Quick start

Upload the two files in `release_assets/` to an A100 Colab session and run:

```python
%run run_moving_fibre_intelligence_l3_v1_9_6_0.py
```

The recorded A100 run took approximately 53.2 minutes.

