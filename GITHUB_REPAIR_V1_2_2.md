# GitHub repair v1.2.2

## Scope

This complete repository snapshot is based on v1.2.1. It repairs documentation
and audit reproducibility without changing frozen experimental code or results.

## Changes

- defines the exact discrete Moving-Fibre F16 action used by v3.2c;
- states explicitly that the argmin is over the frozen six-algorithm class;
- adds general-metric and fibre-critical-point formula qualifications;
- adds a reproducible v3.1e leave-one-radius-out robustness analysis;
- updates claim boundaries, evidence ladder and provenance;
- regenerates repository-wide SHA-256 checksums.

## Safe import

1. Create a branch from the current repository head.
2. Overlay this snapshot at repository root.
3. Run `python audits/formula_audit_v1/recompute_radius_loo.py`.
4. Compare regenerated JSON/CSV with committed files.
5. Verify `provenance/SHA256SUMS`.
6. Commit and create a new tag such as `evidence-v1.2.2-formula-audit`.
   Do not move or overwrite old tags.

The paper source still needs matching edits: `selected directions` in the
abstract, the action definition in Section 6.1, the LOO sentence in Section
5.2, and the restricted-path boundary in Section 6.3.
