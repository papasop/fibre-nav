# GPT-2 LoRA Picard Frozen Confirmation

Version: `v1.6.0`

## Featured result

This release makes the GPT-2 LoRA Picard-vs-AdamW branch the featured
repository result.

| Primary metric | Result |
| --- | ---: |
| Median time-to-equal-loss training speedup | **37.46%** |
| Positive seeds | **5/5** |
| Fixed 600-step compute-time advantage | **1.17%, diagnostic only** |

R4 is the formal frozen five-new-seed confirmation:
`GPT2_LORA_PICARD_V0_2_6_R4_FROZEN_CONFIRMATORY_SUPPORTED`.

R5 is the ten-step target-detection resolution audit:
`GPT2_LORA_PICARD_V0_2_6_R5_TEN_STEP_RESOLUTION_SUPPORTED`. It reuses the same
five seeds as r4, so it improves detection resolution rather than acting as a
second independent confirmation.

## Evidence layout

- `evidence/confirmed/picard_gpt2_lora_r4/`
- `evidence/audits/picard_gpt2_lora_r5_ten_step/`
- `evidence/developmental/picard_gpt2_lora/`
- `external_tests/picard_gpt2_lora/`

The older CIFAR Picard v0.2.6 archive is retained as secondary cross-model
support under `evidence/picard_finetune_v0_2_6/`.

## Boundary

The supported claim is task-specific GPT-2 small rank-4 `c_attn` LoRA evidence
on SHA-256-bound Tiny Shakespeare bytes after a shared 50-step AdamW warm start.
It is not a full-model GPT-2 fine-tuning result, semantic-invariance evidence,
downstream-task transfer, universal optimizer superiority or a global
Picard-flow theorem.
