# GPT-2 LoRA Picard r5 Ten-Step Resolution Audit

This directory archives the r5 measurement-resolution audit for the frozen r4
GPT-2 LoRA Picard result. It checks target loss every 10 steps rather than
every 50 steps, using the same five seeds as r4.

## Featured Audit Result

| Metric | Value |
| --- | ---: |
| Scientific status | `GPT2_LORA_PICARD_V0_2_6_R5_TEN_STEP_RESOLUTION_SUPPORTED` |
| Median time-to-equal-loss speedup | 37.46% |
| Positive seeds | 5/5 |
| Fixed 600-step time advantage | 1.17%, diagnostic |

r5 is the default external reproduction entry point because it resolves target
detection more finely. It is not a second independent new-seed confirmation;
r4 remains the frozen confirmation stage.

## Boundary

The audit is limited to GPT-2 small with rank-4 LoRA on `c_attn`, the frozen
Tiny Shakespeare byte stream, the shared 50-step AdamW warm start excluded from
timing, and the r4 five-seed cohort. It does not establish universal optimizer
superiority or a global Picard-flow theorem.

## Layout

- `code/`: frozen r5 source, README and requirements.
- `raw/`: original source and result ZIPs, plus extracted result JSON for
  inspection.
- `run_summary.json`: machine-readable outcome summary.
- `protocol.json`: repository-level protocol extraction from the frozen
  summary.
- `comparison_r4_r5.json`: machine-readable r4/r5 comparison.
