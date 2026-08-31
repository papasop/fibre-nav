# GPT-2 + LoRA Cached Picard v0.2.6-r1 Quick Development

This is a 30–60 minute A100 development smoke test. It compares AdamW and a cached diagonal-Fisher Picard momentum update on the same GPT-2 small LoRA (`c_attn`, rank 4) parameterization and fixed token schedules.

The run uses a small same-run learning-rate pilot, two new evaluation seeds, 600 steps per arm, synchronized CUDA kernel timing, and a 55-minute hard deadline. It tries WikiText-2 first and logs a public-domain Tiny Shakespeare fallback if the dataset service is unavailable.

The r1 launcher removes an incompatible preinstalled `torchao` before installing PEFT. The benchmark then validates LoRA injection on a local tiny GPT-2 before downloading the full model, uses the current `dtype` API, and suppresses the irrelevant whole-corpus tokenizer-length warning.

`GPT2_LORA_PICARD_V0_2_6_R1_QUICK_POSITIVE_SIGNAL` is only a development signal. It is not the v0.2.6 CIFAR confirmation, an independent GPT-2 confirmation, a universal optimizer claim, or evidence for a global Picard theorem.

Colab: run `picard_gpt2_lora_quick_v0_2_6_r1_colab_launcher.py`, then upload `picard_gpt2_lora_quick_v0_2_6_r1.zip` when prompted.
