# Claim boundary

## Supported statement

Within the frozen pretrained GPT-2 experiment, shared 24-dimensional rank-4
LoRA chart and two declared global response balls, current-response-kernel
updates achieved lower validation loss than both source-frozen updates and
response-budgeted AdamW in all five untouched seeds at each budget. The frozen
confirmation gates passed.

## Required qualifiers

- The comparison is under fixed **global response budgets**.
- The AdamW primary control is **response-budgeted AdamW**.
- The result uses pretrained GPT-2, Tiny Shakespeare, two adapted `c_attn`
  modules and a finite two-coordinate response map.
- The five seeds are the independent units; the two budgets within a seed are
  correlated.
- The current arm uses the identity metric. This result does not confirm an
  inverse-Fisher or metric-preconditioned Picard advantage.

## Not supported

This evidence does not establish:

- superiority to unconstrained AdamW;
- ordinary unconstrained language-model training superiority;
- semantic or downstream-task transfer;
- full-parameter or full-response-map behaviour;
- lower universal per-step computational complexity;
- LLaMA or cross-architecture replication;
- universal optimizer superiority; or
- a global Picard theorem.
