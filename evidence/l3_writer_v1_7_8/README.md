# Moving Fibre Intelligence — L3 Writer v1.7.8

This repository-ready snapshot freezes a single-seed development result for
compositional concept-position writing in GPT-2. Under cyclic concept-code
reassignment, all exact, training-expression, held-out-expression, response
budget, endpoint-KL, mixed-slot margin, and per-view gates passed.

## Frozen result

- Protocol: `MOVING_FIBRE_INTELLIGENCE_L3_MIXED_SLOT_MARGIN_V1_7_8`
- Model: `openai-community/gpt2`
- Seed: `81902`
- Concepts: 4
- Two-slot codes: `[0,0]`, `[0,1]`, `[1,0]`, `[1,1]`
- Cyclic code rotations: 4
- Evaluated concept-code endpoints: 16
- L3 write steps: 520
- Minimum training signed margin: `0.7243118286132812`
- Minimum held-out signed margin: `0.148895263671875`
- Maximum selected endpoint KL: `0.0042860060930252075 < 0.005`
- Maximum response drift: `0.00002288818359375 < 0.002`
- Final status: `all_seeds_pass: true`

This is development evidence, not multi-seed confirmation. See
[`CLAIM_BOUNDARY.md`](CLAIM_BOUNDARY.md).

## Repository layout

```text
src/                              frozen audit, config, requirements
colab/                            notebook-safe launcher
evidence/v1_7_8/                  summary, console log, full seed record
release_assets/                    original Colab launcher and experiment ZIP
CLAIM_BOUNDARY.md                 supported and unsupported claims
REPRODUCE.md                      Colab reproduction instructions
SHA256SUMS.txt                    integrity manifest
```

## Reproduce in Colab

Upload both files in `release_assets/` and run the launcher. An NVIDIA A100 run
took approximately 43 minutes in the frozen record.

## Suggested development tag

`mfi-l3-writer-v0.1.0-dev`

Choose and add a license before making the repository public.
