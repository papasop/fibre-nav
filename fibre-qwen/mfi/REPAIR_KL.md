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

## Successor: response and KL correction

Candidate 1 repaired the known failed WRITEs but could accumulate response drift
near 0.02 during OVERWRITE. `repair_response.py` adds a least-squares inward
response correction when response drift reaches half its budget. It aims to halve
the current signed response residual in the local linear model. The KL correction
then acts within the response tangent space. These are local approximations; only
the existing finite gates authorize committing an operation.

The successor direction is not strictly response-null: its normal component
intentionally retracts accumulated drift. All parameters remain in the same
restricted LoRA-B chart. No thresholds or iteration limits were relaxed. Candidate
1 and its protocol are retained alongside the successor, rather than overwritten.

`verify_repair2.py` froze new seeds 84301, 84317 and 84329 before execution. Its
development comparison uses the archived original scores; the new seeds get fresh
paired original/successor runs. The previously successful development seed 84031
has an explicit no-regression gate. The report's development/original entry has
zero new runs (accuracy null); use the archived baseline for that comparison.

Frozen candidate 1 commit: a5fcb8b2aae27f5ae44166a9b8ca9857fedaf072.
Frozen candidate 2 commit: 65f4405798094f305b1b15455da0c27c68ac0ba4.
Candidate 2 was designed after observing development failures in candidate 1;
its development results are not prospective evidence. Its separately selected
three new seeds were not used for tuning. Both protocols retain raw task outputs
and rollback checks and share the same fixed task items and scoring rule.
