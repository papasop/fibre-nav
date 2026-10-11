# B4 full XOR V2 — protocol only, results pending

**Status: NOT YET RUN ON GPT-2.** This directory contains the frozen A100 follow-up protocol, not positive evidence.

Use `GPT2_B4_FULL_XOR_A100_V2.py` in one Colab A100 cell. Source documentation and tests are in `source/`. The ZIP is the identical source package embedded in the PY.

Three fixed seeds, four XOR cases, five phases per case, maximum 30,240 steps. Normal and substituted writebacks must flip output; wrong-read counterfactuals execute actual writebacks. Fresh expression checks are excluded from optimization/selection. Drive backup every five steps, automatic verified V2-only resume. First run starts from zero B; V1 results are not resume inputs.

Previously completed evidence: [B4 V1 minimal development](../../evidence/gpt2_b4_causal_xor_a100_v1/README.md).

Local checks cover protocol logic and synthetic workflow/resume only. Do not mark B4 complete based on those checks. Real GPU results and independent replication are separate.
