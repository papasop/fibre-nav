# GPTW Response-Fibre Evidence v1.1.1

This is a documentation and protocol-completeness correction for
`evidence/gptw_response_fibre_v1_1_0/`.

No numerical result or frozen gate changed. Documentation/protocol-completeness
correction only.

## Scope

The authoritative frozen engine, per-seed JSON files, report, decision record
and v1.1.0 `SHA256SUMS` remain in the original v1.1.0 archive. This revision
adds an explanatory boundary for the v2 current-versus-source-fixed audit and
records protocol constants that were present in source code but absent from the
archived `protocol.json`.

## v2 Interpretation Correction

In the v2 codeword cohort, the two arms are defined differently with respect to
the response Jacobian at an interior path node:

```text
R(theta + h v) - R(theta) = h J v + (h^2 / 2) H[v, v] + O(h^3)
J v_current approx 0  =>  e(h) = O(h^2)
J v_fixed   != 0      =>  e(h) = O(h)
```

Consequently, `alpha_current approx 2` and `alpha_fixed approx 1` are
numerical-correctness checks implied by the arm definitions and finite
Taylor expansion. The same applies to the slope gap. They are not independent
discoveries.

The substantive empirical v2 result is the finite-radius separation under the
frozen gates:

| Quantity | Frozen v2 value |
|---|---:|
| Seeds passing | 6/6 |
| Interior nodes passing | 18/18 |
| Finest-radius fixed/current cost ratio | 8.60-25.70 |
| Principal angle between current and source-fixed directions | 0.0161-0.0713 rad |
| Maximum path-response error | 6.7e-16 |
| Frozen path-response error gate | 2e-4 |

The archived slope ranges remain useful diagnostics:

| Diagnostic | Range |
|---|---:|
| `alpha_current` | 1.9986-2.0084 |
| `alpha_fixed` | 1.0849-1.2151 |

They should be described as consistency checks for the declared v2 arm
construction, not as the main empirical claim.

## v2 Versus v3

The v2 codeword cohort and v3 natural-text cohort are distinct and should not be
merged when reporting numbers.

| Cohort | Passing result | Substantive finite-radius evidence |
|---|---|---|
| v2 codeword | 6/6 seeds, 18/18 nodes | finest-radius cost ratio 8.60-25.70; principal angle 0.0161-0.0713 rad; active-J residual amplification about 1.4e4-1.4e5; path-response error at numerical precision |
| v3 natural text | 8/8 seeds, 32/32 nodes | finest-radius cost ratio 8.87-35.56; active-J residual amplification 1.41e7-1.12e8; maximum path-response error about 1.08e-15 |

The v3 natural-text experiment is part of the paper-facing claim set. It is not
a hidden result and does not require a new paper section; paper edits should
only ensure that the reported v3 numbers and repository cohort refer to the
same frozen queue. It confirms that the result is not confined to the original
codeword prompts, not semantic invariance or full-model GPT-2 behaviour.

## Files

- `V2_PROTOCOL_COMPLETENESS_ADDENDUM.md` records the v2 constants, response
  definition and ridge solve formula.
- `CLAIM_BOUNDARY.md` restates the corrected claim boundary.
- `SOURCE_MANIFEST.md` lists the frozen source materials used by this
  documentation revision.
- `SHA256SUMS` hashes the v1.1.1 documentation files only.
