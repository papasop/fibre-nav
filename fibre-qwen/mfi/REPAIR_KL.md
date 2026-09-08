# KL-aware writer candidate

The old writer projects the bit-target gradient against the three response rows.
It checks KL only after proposing a finite step. The two failed development charts
reach KL near 0.01 with target margins only 0.218 and 0.811, so shorter steps along
the same direction cannot finish the write within the frozen iteration budget.

`repair_kl.py` is a separate candidate; `core.py` and `qwen_l1.py` remain unchanged.
It calculates the full anchor KL gradient against the original frozen log-probability
reference. When current KL is at least half its budget, it projects that gradient
into the response tangent space, removes outward motion in this direction, and
adds an inward component of 0.25 times the original projected target-gradient norm.
The resulting direction is normalized and tested using the original line search.

This first-order direction is not itself a safety certificate. The existing finite
response/KL/margin gates and transactional rollback remain authoritative, including
rollback on exceptions. Model, rank, anchors, readout, response/KL thresholds,
margin, 24 iterations, step norm and eight backtracks remain unchanged. More gradient
work per iteration is required, so runtime must be measured rather than assumed equal.

`verify_repair.py` freezes three development and three new chart seeds, runs both
original and candidate writers, and evaluates the same preference task independently.
All failed/skipped stages remain in its task denominator. Task scores do not influence
updates, line search or early stopping. A paired no-regression gate accompanies the
all-six repaired-program gate. New-seed results are prospective for this candidate;
reusing those seeds in later tuning would make them development evidence.

The candidate is not automatically adopted by the HF app. Archived baseline evidence
and the deployed protocol remain intact until a verified successor is selected.

Run in the pinned environment:

```sh
HF_HUB_OFFLINE=1 python verify_repair.py --out results/kl_repair_NEW
```

Offline mode requires the previously downloaded pinned Qwen model. Protocol and
source snapshots are written before evaluation; exit 2 preserves failed gates.
