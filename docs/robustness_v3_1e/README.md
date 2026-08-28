# v3.1e leave-one-radius-out robustness

This directory contains a post-confirmation sensitivity analysis of the
four-radius v3.1e scaling fit, reconstructed from archived per-seed JSON by:

```bash
python audits/formula_audit_v1/recompute_radius_loo.py
```

Each fit deletes one radius, refits every comparable seed, and reports the
median exponent across seeds. This was not a prospectively frozen gate. It
tests whether the confirmed moving/fixed separation is carried by one radius.

The largest moving-arm median shift is approximately 0.048; the smallest LOO
moving-minus-fixed median separation is approximately 0.627, remaining above
the original 0.25 separation threshold.
