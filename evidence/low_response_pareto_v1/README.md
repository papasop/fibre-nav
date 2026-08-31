# Prospective low-response Pareto audits

This directory archives two separately frozen audits of held-out learning
utility under finite response budgets at recorded optimizer nodes.

The common decision structure is:

1. record optimizer nodes and proposals;
2. construct instantaneous-kernel and control directions;
3. select a finite step scale on calibration data under the same frozen
   response budgets for every arm;
4. score the selected scale once on disjoint held-out data;
5. compare the areas under the four-budget utility frontiers.

The successful instantaneous-kernel arm is a counterfactual projection. These
audits do not show that AdamW naturally follows a response fibre.

| Audit | Prospective result | Status |
| --- | --- | --- |
| Reduced ResNet v4.6 | 7/8 supporting seeds | Supported |
| GPT-2 native-LoRA-B v1-r1 | 8/8 seeds; 24/24 noninitial nodes | Confirmed within the declared protocol |
| GPT-2 native-LoRA-B R2 strict | 8/8 seeds completed; 6/8 complete-control supporting seeds; 7/8 positive same-kernel seed contrasts; 20/24 positive noninitial nodes | Strict-control confirmed within the declared protocol |

The architectures use different response maps, parameter charts and effect
scales. The shared claim concerns ordering under the frozen protocol, not
numerical equality across architectures.

See `CLAIM_BOUNDARY.md` and the per-audit claim boundaries before citing any
result.
