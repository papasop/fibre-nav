# R17d: Pythia-160M metric-constrained Onsager confirmation

Authoritative status: `R17D_METRIC_CONSTRAINED_ONSAGER_CONFIRMED`.

Five untouched seeds tested the R17c-frozen multiplier 1.15.  The current
metric-constrained Onsager arm beat current-kernel projected AdamW in 4/5
seeds and source-frozen metric Onsager in 5/5.  Median loss margins were
+0.00135338 versus AdamW and +0.01599610 versus source-frozen.  Median accuracy
difference versus AdamW was -0.390625 percentage points, within the frozen
-0.5-point noninferiority gate.  All frozen gates passed.

`raw/pythia_r17d_results.zip` is authoritative. `results/` is its extracted
inspection copy. `code/` contains the exact aggregate runner, per-seed worker,
frozen protocol, launcher and shared engine.

Scope is limited to pretrained Pythia-160M, SST-2 learning, a disjoint frozen
AG News response map, one 32-dimensional LoRA chart and one global response
budget.  This is not universal optimizer superiority or a physical law.
