# GPT-2/native-LoRA-B Pareto R2: interpretation

## Frozen question

At recorded AdamW nodes, does the task-related projection of the recorded
update into the current response kernel achieve greater held-out utility under
the same response budgets than a calibration-selected best-of-16 equal-norm
random direction in that same kernel?

## Result

- Decision: `GPT2_LORA_LOW_RESPONSE_PARETO_STRICT_CONFIRMED`
- Complete cohort: 8/8 seeds, 24 noninitial nodes
- Complete-control supporting seeds: 6/8
- Positive seed-level same-kernel contrasts: 7/8
- Positive node-level same-kernel contrasts: 20/24
- Bootstrap 95% CI for the mean seed contrast: `[0.0296015, 0.1282980]`
- Exact two-sided sign-flip p-value: `0.03125`
- GPU: NVIDIA A100-SXM4-40GB

The best-of-16 random control selected candidates using calibration AUC only;
held-out AUC did not participate in selection. It attained the full scale at
all four budgets at every primary node. The result therefore cannot be reduced
to the random control having a narrower feasible scale.

## Claim boundary

The result distinguishes task-related direction selection from generic
membership in the current response kernel in a finite artificial GPT-2
rank-2 native-LoRA-B task. It does not show that AdamW naturally follows the
fibre or that response-fibre capacity predicts or determines subsequent AdamW
learning. PC1-PC4 remain separate negative boundaries.

R2 is a prospective protocol with new seeds. It does not overwrite, repair or
retroactively strengthen the historical R1 decision.
