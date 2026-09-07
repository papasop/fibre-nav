# GPT-2 Fibre Memory L2 v1.5.6 — one-seed worst-margin smoke

This development run fixes seed `81801` and tests L1+L2 only. Training remains
identical to v1.5.5. The causal overwrite receives 520 steps. The semantic
control gate now compares the worst held-out signed margin against exact-only,
which measures the weakest item instead of hiding one failure in a median.

Held-out prompts remain excluded from training and checkpoint selection. The
prompts, split, controls, response budget, KL gate, and overwrite protocol are
The A reserve KL remains `0.0038`; initial write remains 360 steps. Held-out
prompts remain excluded from training and checkpoint selection. L3/L4 are not run.

Run `run_fibre_memory_colab_l2_v1_5_6.py` in Colab and upload
`gpt2_fibre_memory_l2_v1_5_6.zip` when prompted. Expected A100 runtime is about
25–30 minutes. The launcher downloads `fibre_memory_l2_results_v1_5_6.zip`.

This is a development experiment, not confirmation.
