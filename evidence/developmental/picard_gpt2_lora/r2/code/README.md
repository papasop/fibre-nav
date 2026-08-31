# GPT-2 + LoRA Cached Picard v0.2.6-r2 Warm-Chart Development

This is a sub-hour A100 mechanism test. Both arms first execute the identical 50-step AdamW warm-start. From that same nonzero LoRA state, the script constructs the frozen diagonal Fisher, discards optimizer state, and compares fresh AdamW with cached Picard on identical token schedules.

Picard uses fused `torch._foreach_*` updates and searches `0.01/0.03/0.1/0.3`. The report certifies identical warm-start validation losses and records the post-warm Fisher range. The run retains two evaluation seeds, 600 timed steps per arm, synchronized CUDA timing, and a 55-minute hard deadline.

The launcher removes incompatible preinstalled `torchao`; the benchmark validates LoRA injection on a local tiny GPT-2 before downloading the full model.

`GPT2_LORA_PICARD_V0_2_6_R2_WARM_CHART_POSITIVE_SIGNAL` is only a development signal. The shared AdamW warm-start is excluded from comparison timing. This is not independent confirmation or a universal optimizer claim.

Colab: run `picard_gpt2_lora_quick_v0_2_6_r2_colab_launcher.py`, then upload `picard_gpt2_lora_quick_v0_2_6_r2.zip` when prompted.
