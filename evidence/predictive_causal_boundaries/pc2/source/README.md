# GPTW-PC2: optimizer access to predictive response-fibre capacity

PC1 found that current-kernel capacity correlated with future AdamW learning
but did not add sufficient seed-blocked predictive power beyond ordinary node
statistics. PC2 does not revise that negative decision. It asks a different
mechanistic question:

> Does capacity become predictive when the future optimizer proposals are
> explicitly projected into the moving current response kernel?

At three recorded nodes per seed, the audit estimates current-, source- and
signed-permuted-kernel capacities using disjoint scale-selection and
capacity-scoring templates. From the same node it then launches four matched
four-step branch-local AdamW interventions:

1. unprojected AdamW;
2. AdamW proposals reprojected into the current kernel at every branch step;
3. AdamW proposals projected into the node's frozen source chart;
4. AdamW proposals projected into a deterministically signed-permuted current
   chart at every branch step.

The prediction target is future loss improvement on a third, disjoint template
set. The primary analysis is leave-one-seed-out ridge prediction after
controlling for ordinary node variables.

## Interpretation

A positive result would support the distinction between latent response-fibre
capacity and optimizer access to that capacity. It would not repair PC1, prove
that ordinary AdamW naturally follows the fibre, establish generalization, or
define a universal representation measure.

## Colab CPU

Upload the standalone launcher, run it, then upload this ZIP when prompted:

```python
%run COLAB_LAUNCHER_GPTW_PC2_CPU.py
```

No GPU is required. The frozen hard limit is two hours.
