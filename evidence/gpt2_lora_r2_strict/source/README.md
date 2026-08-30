# GPTW low-response Pareto GPU V1-R2 strict audit

R2 preserves the R1 prompts, budgets, alpha grid, layers, rank and optimizer,
but uses eight new seeds and adds the missing best-of-16 equal-norm random
direction inside the true current response kernel. Candidate selection uses
calibration AUC only; held-out AUC is untouched until scoring. Far-time controls
cannot duplicate the source control at a primary node. Inference is seed-level.

The package also supports NumPy before and after removal of `np.trapz`, and a
timeout produces a recoverable partial archive with `INCOMPLETE_TIMEOUT` rather
than a scientific decision.

Run the separate Colab launcher in a GPU runtime and upload this ZIP. A100 is
optional. Allow up to two hours.
