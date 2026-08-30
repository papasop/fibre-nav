# GPT-2 native-LoRA-B low-response Pareto audit

Decision: `GPT2_LORA_LOW_RESPONSE_PARETO_CONFIRMED` within the declared
protocol.

- prospective seeds: 86841-86848;
- supporting seeds: 8/8;
- audited noninitial nodes: 24;
- current positive and beats every declared control: 24/24;
- pooled median AUC:
  - current kernel: 0.11858;
  - source kernel: 0.03341;
  - half-path time-shifted kernel: 0.03641;
  - recorded AdamW: 0.00851;
  - signed permutation: 0.00853.

The standard GPT-2 backbone is frozen. The declared trainable chart contains
rank-2 native LoRA-B coordinates in layers 10 and 11 (9,216 parameters), with
the random frozen LoRA-A factor included in the seed axis. The artificial
color-code task is expressed through disjoint natural-language templates and
does not establish general semantic invariance.

The parent engine used by the eight-seed wrapper is archived in this directory
as `gptw_lora_low_response_pareto_cpu.py`. It is the original source file from
the frozen input ZIP, not a reconstructed replacement.

The historical Colab launcher below expects an author-supplied ZIP upload and
is preserved for provenance:

```python
%run COLAB_LAUNCHER_GPTW_LORA_8SEED_CPU.py
```

For a zero-upload same-cohort external reproduction, use:

```bash
python external_tests/gpt2_lora_pareto/COLAB_ONE_CLICK_GPT2_LORA_PARETO.py \
  --source-root . --verify-archived
python external_tests/gpt2_lora_pareto/COLAB_ONE_CLICK_GPT2_LORA_PARETO.py \
  --source-root . --smoke
python external_tests/gpt2_lora_pareto/COLAB_ONE_CLICK_GPT2_LORA_PARETO.py \
  --source-root . --full
```
