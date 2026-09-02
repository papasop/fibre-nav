# Fibre-Qwen R23c — general moving-fibre compute-core smoke test

This development benchmark loads `Qwen/Qwen3-0.6B`, inserts rank-4 LoRA adapters into the final two attention blocks, and compares three frozen arms on disjoint general-instruction and response-probe sets:

1. `moving_current_kernel`: recompute `DR(theta)` every step and project the Adam direction into its numerical kernel;
2. `source_frozen_kernel`: reuse `DR(theta_0)`;
3. `budgeted_lora_adamw`: use the unprojected AdamW direction under the same finite global response budget.

The response vector consists of frozen A/B logit-margin coordinates on prompts excluded from the learning objective. Every finite proposal is backtracked against the actual response distance from the source model. This is a one-seed engineering smoke test, not evidence of general capability, continual learning, or moving-fibre superiority.

Run `colab_launcher_r23c_r2.py` in Colab with an A100. R23c-r2 uses FP32 model/LoRA forward passes and FP64 response aggregation, and adds hard gates for measurable response drift, distinguishable budgets, moving Jacobian row space, and projected-direction residual. The launcher validates the package and entry point, streams the complete child log, and fails closed on a missing or mismatched summary. Expected runtime is roughly 5–15 minutes on an A100 after download.
