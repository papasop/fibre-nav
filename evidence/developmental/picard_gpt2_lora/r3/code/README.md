# GPT-2 + LoRA Picard v0.2.6-r3 Metric-Exponent Screen

This is the final sub-hour direct-transfer development screen. It compares `g/G^alpha` for `alpha = 0, 0.25, 0.5, 1` after the same 50-step shared AdamW warm-start.

Every exponent searches learning rates `0.3/0.6/1.0/2.0`. Nonfinite candidates or candidates whose update norm exceeds the parameter norm are ineligible. The best stable pilot geometry is evaluated against AdamW on two new seeds. Fixed-budget speed is diagnostic; the decisive development gate is positive time-to-equal-loss speedup with noninferior loss.

The launcher removes incompatible preinstalled `torchao`; the benchmark validates LoRA injection on a local tiny GPT-2 before downloading the full model.

If no geometry passes, the status is `GPT2_LORA_PICARD_V0_2_6_R3_DIRECT_TRANSFER_NOT_SUPPORTED` and direct GPT-2 LoRA transfer should stop rather than continue tuning.

Colab: run `picard_gpt2_lora_quick_v0_2_6_r3_colab_launcher.py`, then upload `picard_gpt2_lora_quick_v0_2_6_r3.zip` when prompted.
