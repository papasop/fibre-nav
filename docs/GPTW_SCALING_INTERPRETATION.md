# GPTW Scaling Interpretation

This note calibrates the interpretation of the GPTW v2 and v3
current-versus-source-fixed audits. It changes documentation only. No frozen
engine, result JSON, report, gate or decision is modified.

## Taylor Expansion

For a response map `R` at a path node,

```text
R(theta + h v) - R(theta) = h J v + (h^2 / 2) H[v, v] + O(h^3).
```

The GPTW current arm is reprojected into the current response kernel at each
audited node, so `J v_current approx 0`. Its leading response error, and hence
the normal correction cost computed from that error, is therefore expected to
scale like `O(h^2)`.

The source-fixed arm replays a direction from the source geometry at later
nodes. It retains a nonzero active-Jacobian component, so `J v_fixed != 0` and
the leading correction is expected to scale like `O(h)`.

Thus the 2-versus-1 exponent split is analytically forced by the arm definitions
and retained as a numerical-correctness check. The slope gap should not be
presented as an independent geometric discovery.

## Nontrivial Evidence

The empirical content of the GPTW current-versus-source-fixed branch is the
finite-radius magnitude and prospective replication under frozen gates.

| Cohort | Evidence |
|---|---|
| v2 codeword | 6/6 seeds, 18/18 nodes; finest-radius fixed/current correction-cost ratio 8.60-25.70; principal angle 0.0161-0.0713 rad; active-J residual amplification about 1.4e4-1.4e5; path-response error at numerical precision |
| v3 natural text | 8/8 seeds, 32/32 nodes; finest-radius fixed/current correction-cost ratio 8.87-35.56; active-J residual amplification 1.41e7-1.12e8; maximum path-response error about 1.08e-15 |

V3 is a disjoint frozen natural-text prompt cohort. It supports that the
current-versus-source-fixed separation is not confined to the original codeword
prompts. It does not establish full-model GPT-2 behaviour, semantic invariance,
downstream-task improvement, ordinary SGD/Adam behaviour, a global fibre bundle
or arbitrary-path variational optimality.
