# GPT-2/native-LoRA-B low-response Pareto R2 strict audit

This directory contains the complete source and result snapshot for the
prospective R2 strict-control audit.

- `source/`: executable engine, Colab launcher, frozen protocol and requirements
- `results/`: report, protocol copy, 8 per-seed records, node metrics and log

The primary new control is a calibration-selected best-of-16 equal-norm random
direction inside the true current response kernel. R1 remains historically
separate under `../gpt2_lora_b_v1/`.

Run the repository-root verifier after import:

```bash
python verify_gpt2_lora_pareto_r2.py
```
