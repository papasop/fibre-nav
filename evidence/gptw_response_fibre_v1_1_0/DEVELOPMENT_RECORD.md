# Development and decision record

## v1-r1: formal negative, scientifically informative

The first dependency-robust implementation completed eight seeds. Selected tangents beat no-tangent, sign reversal and best-of-16 random controls in 8/8 and shuffled targets in 7/8, but every instance failed the frozen finite-difference/Jacobian gate. The reported relative error divided by `||Jv||`, even though kernel projection intentionally makes `Jv` approximately zero. The frozen decision remained negative.

## v1-r2: execution failure

The planned float64 repair stopped at seed 24742 before producing scientific output because random controls and a ridge identity matrix retained float32 defaults. The original log is retained under `development/`. No threshold, seed or scientific hypothesis was changed.

## v1-r2a: same-seed precision repair

All generated controls were made dtype-consistent. The derivative audit used central differences at `1e-2`, `3e-3` and `1e-3`, normalized by `||J|| ||v||` rather than near-zero `||Jv||`. Seven of eight original seeds passed every gate; this validated the repair but was not treated as an independent confirmation.

## v1-r3: prospective confirmation

The complete r2a protocol was frozen and only the seeds were changed from 24742-24749 to the disjoint cohort 25742-25749. All eight new seeds passed every effect, leakage, kernel and precision gate. Frozen requirement: at least six of eight.

## v2: moving-fibre extension

Six further disjoint seeds 26742-26747 were used. Each constructed path contained three audited interior nodes. All 18 nodes and all six instances passed the preregistered current-versus-source-fixed realizability gates. This supports the restricted moving-fibre mechanism, subject to `CLAIM_BOUNDARY.md`.

## v3: prospective natural-English confirmation

The architecture, rank-2 LoRA-B domain, current-versus-source-fixed construction and 55-minute hard limit were retained. Synthetic codeword prompts were replaced by two frozen, disjoint natural-English sets: development continuations selected the tangent direction, while confirmation continuations defined the response map. Eight new seeds 38171-38178 and four interior nodes per seed were frozen. The prospective gate required at least six supporting instances, at least three passing nodes per supporting instance, current slope at least 1.70, fixed slope at most 1.40, slope gap at least 0.50, active-residual ratio at least 100, finest-radius correction-cost ratio at least 3, current-kernel residual at most 1e-5, and path response error at most 1e-10.

All eight instances and all 32 nodes passed. The run completed in 125.2 seconds on the reported A100 environment. No threshold, prompt, seed or decision rule was changed after observing results. V3 is a prospective confirmation beyond the codeword prompt construction, not a full-model, semantic-generalization, downstream-performance or global-variational result.
