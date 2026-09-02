# Fibre-Qwen R23d — frozen five-seed confirmation

R23d freezes the complete R23c-r2 configuration and tests it on five previously unused seeds. It loads `Qwen/Qwen3-0.6B`, inserts rank-4 LoRA adapters in the final two attention blocks, and compares:

1. `moving_current_kernel`: recompute `DR(theta)` every step and project into its numerical kernel;
2. `source_frozen_kernel`: reuse `DR(theta_0)`;
3. `budgeted_lora_adamw`: unprojected AdamW under the same finite global response budget.

The learning set and six disjoint frozen response coordinates are unchanged from R23c-r2. The two response budgets, step count, step norm, precision floor, and all gates are frozen. No tuning occurs in R23d.

Run `run_fibre_qwen_r23d_colab.py` in a fresh Colab A100 session and upload `fibre_qwen_r23d.zip` when prompted. Expected benchmark time after model download is approximately 15–25 minutes. The launcher streams progress, validates the final protocol, creates `fibre_qwen_r23d_results.zip`, and downloads it automatically.

Passing R23d confirms only the frozen five-seed ordering in this Qwen3-0.6B, tiny authored-data, restricted-LoRA setting. It is not evidence of broad capability, personalization, continual learning, safety, deployment readiness, or universal optimizer superiority.
