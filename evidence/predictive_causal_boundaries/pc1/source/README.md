# GPTW-PC1: predictive response-fibre capacity audit

This frozen CPU audit asks whether a locally measured response-budget capacity
predicts **subsequent realized learning**, rather than merely redescribing the
same local utility curve.

At four recorded nodes per seed, candidate step scales are selected on one
template set. The resulting low-response utility is scored on a second,
disjoint template set to estimate

\[
\widehat{\mathcal C}_\varepsilon(\theta_t).
\]

The prediction target is the loss improvement produced by the next six actual
AdamW steps on a third, disjoint future-evaluation template set. No
future-evaluation value enters the capacity estimate or scale selection.

The primary analysis uses leave-one-seed-out prediction. A fixed ridge model
containing ordinary state variables (step, training loss, capacity-selection
loss, gradient norm, update norm, and full-update response cost) is compared
with the same model plus current-kernel capacity. Source-kernel and
signed-permuted capacities are frozen structural controls.

## Scientific role

This is a prospective test of whether response-fibre capacity is a missing
predictive state variable. A positive result would extend the paper's local
behaviour--learning separation; it would not prove a general theory of
representation, explain SGD globally, or establish population generalization.

## Run in Colab CPU

Upload this ZIP and the launcher when prompted, then run:

```python
%run COLAB_LAUNCHER_GPTW_PC1_CPU.py
```

No GPU is required. The frozen hard limit is two hours. Runtime depends on the
Colab CPU and model-download speed.

## Output

The launcher downloads `gptw_pc1_predictive_capacity_results.zip`, containing:

- `report.json`;
- `protocol.json`;
- eight per-seed JSON records;
- `node_metrics.csv`;
- `loo_predictions.csv`;
- the execution log.

Interpret the prospective decision and every frozen gate. Do not promote a
failed or ineligible result by changing thresholds after inspection.
