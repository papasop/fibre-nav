# Moving Fibre Intelligence — L3 shared-category writer v1.8.1

This snapshot freezes a single-seed development result for a shared
category-conditioned writer in GPT-2. It is a partial result, not confirmation.

## Frozen protocol

- Protocol: `MOVING_FIBRE_INTELLIGENCE_L3_SHARED_CATEGORY_WRITER_V1_8_1`
- Model: `openai-community/gpt2`
- Seed: `82001`
- Categories: city, country, continent, fruit
- Training concepts: 16
- Whole-concept holdouts: 8
- L3 write steps: 520
- Category move scale: 4.0
- Runtime in the frozen A100 record: 3998.78 seconds

## Result

- Writer endpoints eligible: pass
- Exact access: pass
- Training-expression access: pass
- Shared router training accuracy: 16/16
- Shared router held-out accuracy: 7/8
- Category geometry held-out accuracy: 7/8
- Category moves eligible: 24/24
- Beats shuffled-category control: pass
- Beats no-graph control: pass
- Positive held-out category margin: fail; minimum `-1.4898548126220703`
- Within/between ratio gate: fail; observed `1.0561434734775528`
- Beats pair-only control: fail
- All writer held-out expressions: fail; Japan reached 5/6
- Final status: `all_seeds_pass: false`

The failed whole-concept route was `grape`, predicted as `continent` rather than
`fruit`. This record motivates v1.8.2 full-description mean pooling, increased
category move scale and additional writer steps.

See `CLAIM_BOUNDARY.md` before citing this snapshot.
