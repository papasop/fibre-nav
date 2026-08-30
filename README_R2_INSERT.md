## Strict GPT-2/native-LoRA-B low-response Pareto confirmation

The prospective GPU R2 audit adds the missing direction-specific control: a
calibration-selected best-of-16 equal-norm random direction inside the same
instantaneous response kernel. Six of eight new seeds passed the complete
frozen control family. The current-minus-random-current-kernel seed contrast was
positive in seven of eight seeds and twenty of twenty-four noninitial nodes;
the bootstrap 95% confidence interval for the mean seed contrast was
`[0.0296, 0.1283]` (`p=0.03125`, exact two-sided sign flip).

This supports task-specific low-response Pareto advantage beyond generic
current-kernel membership. It does not establish natural AdamW navigation,
prediction of future optimizer dynamics, full-parameter GPT-2 geometry, broad
semantic learning, or a universal representation theory. Historical R1 files
and decisions remain unchanged.

Evidence and one-click source are under:

```text
evidence/low_response_pareto_v1/gpt2_lora_b_r2_strict/
```
