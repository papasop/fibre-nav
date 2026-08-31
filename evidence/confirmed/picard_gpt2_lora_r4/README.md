# GPT-2 LoRA Picard r4 Frozen Confirmation

This directory archives the frozen r4 GPT-2 LoRA Picard-vs-AdamW confirmation.
It is the repository-backed confirmation stage for the Picard GPT-2 LoRA
branch.

## Result

| Metric | Value |
| --- | ---: |
| Scientific status | `GPT2_LORA_PICARD_V0_2_6_R4_FROZEN_CONFIRMATORY_SUPPORTED` |
| Positive seeds | 5/5 |
| Median time-to-equal-loss speedup | 34.38% |
| Median fixed 600-step time advantage | 1.66%, diagnostic |
| Median final validation-loss delta | -0.02147 |

## Boundary

The protocol uses GPT-2 small with rank-4 LoRA on `c_attn`, SHA-256-bound Tiny
Shakespeare bytes, a shared 50-step AdamW warm start excluded from comparison
timing, and frozen r3-selected Picard hyperparameters. It is task-specific
evidence, not a universal optimizer comparison, a full-model GPT-2 result, or
a global Picard-flow theorem.

## Layout

- `code/`: frozen r4 source, README and requirements.
- `raw/`: original source and result ZIPs, plus extracted result JSON for
  inspection.
- `run_summary.json`: machine-readable outcome summary.
- `protocol.json`: repository-level protocol extraction from the frozen
  summary.
