# Neural Fibre Geometry

Repository evidence archive for *The Geometry of Functional Freedom in Neural Networks*.

This tree separates confirmed finite evidence, developmental follow-up evidence,
claim boundaries, and provenance. The current archive preserves the CNER-F v16
confirmation, the Moving-F16 v3.0b confirmation, and developmental moving-fibre
experiments v3.1a through v3.1c.

## Result status

| Result | Status | What can be claimed |
|---|---|---|
| F16 v16 | Confirmed | Frozen six-algorithm-family restricted CNER-F ordering. |
| Moving-F16 v3.0b | Confirmed | The restricted ordering persists under the moving Fisher metric. |
| Moving-fibre v3.1a | Developmental | Quick positive result for the current response kernel. |
| Moving-fibre v3.1b | Mixed/failed gate | Deep paths still show signal, but the complete preregistered gate did not pass. |
| Realisability-cost v3.1c | Candidate supported | Realizable online moving-fibre paths and fixed-chart replay show different cost scaling. |

## Evidence layout

```text
evidence/
  confirmed/
    f16_v16/
    moving_f16_v3_0b/
  developmental/
    moving_fibre_v3_1a/
    moving_fibre_v3_1b/
    realisability_cost_v3_1c/
  boundaries/
    gpt2/
```

Each evidence stage keeps its executable code, extracted results, raw result
archives, and stage README together:

```text
evidence/<status>/<stage>/
  code/
  results/
  raw/
  README.md
```

## Confirmed evidence

- `evidence/confirmed/f16_v16/`: restricted CNER-F v16 confirmation in the
  frozen TinyCNN/MNIST chart.
- `evidence/confirmed/moving_f16_v3_0b/`: Moving-F16 restricted ordering with
  pointwise recomputation of the output-Fisher pullback and capacity dual norm.

## Developmental evidence

- `evidence/developmental/moving_fibre_v3_1a/`: quick online moving response
  fibre candidate.
- `evidence/developmental/moving_fibre_v3_1b/`: deeper moving response fibre
  candidate; not supported as a full gate because the fixed-chart action
  separation gate failed.
- `evidence/developmental/realisability_cost_v3_1c/`: step-scaling candidate
  for retraction Fisher length and tangent residual separation.

## Boundaries

`evidence/boundaries/gpt2/` is reserved for GPT-2 and language-model transfer
boundary records. GPT-2 supports a Fisher-distance phenomenon, but does not yet
support capacity-weighted CNER or local variational minimality. GPT-2 development
code is intentionally not included in this main evidence trunk.

See:

- `docs/EVIDENCE_LADDER.md`
- `docs/CLAIM_BOUNDARIES.md`
- `evidence/boundaries/gpt2/STATUS.md`

## Provenance

Integrity and provenance records live under `provenance/`.

- `provenance/PROVENANCE.md`
- `provenance/SHA256SUMS`

Run this from the repository root to verify the current tracked evidence tree:

```bash
shasum -a 256 -c provenance/SHA256SUMS
```

## License

No open-source license is granted unless an explicit license is added by the
repository owner. See `LICENSE`.
