# GPTW-PC3: dynamic persistent response-fibre capacity

PC1 found that instantaneous current-kernel capacity did not independently
predict six-step recorded AdamW learning. PC2 found that the same scalar did
not become predictive when current-kernel access was imposed. Those negative
decisions remain unchanged.

PC3 prospectively tests a different state description:

\[
\mathcal S_t=
\left(
\mathcal C_t,
\frac{\mathcal C_t-\mathcal C_{t-6}}{s_t-s_{t-6}},
\frac{\|P_t-P_{t-6}\|_F}{s_t-s_{t-6}}
\right),
\]

where (s) is accumulated AdamW path length and (P_t) is the response-row
space projector (equivalently, its change has the same Frobenius norm as the
kernel-projector change). Only present and past quantities enter the predictor.

Sixteen new seeds are audited at steps 12 and 18, producing 32 seed-blocked
predictions of the next six recorded AdamW steps. Capacity-scale selection,
capacity scoring and future evaluation use disjoint template sets.

Primary comparisons:

1. ordinary node baseline;
2. baseline plus static current capacity;
3. baseline plus dynamic current state;
4. baseline plus dynamic source-capacity history;
5. baseline plus current capacity and deterministically seed-mismatched history.

A positive result supports a dynamic predictive state within this protocol;
it does not revise PC1/PC2, prove optimizer naturality, or establish general
representation or generalization theory.

## Colab CPU

Upload the launcher and run:

```python
%run COLAB_LAUNCHER_GPTW_PC3_CPU.py
```

Upload the ZIP when prompted. No GPU is required; hard limit is two hours.
