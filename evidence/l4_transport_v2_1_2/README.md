# L4 cross-fibre transport: large-atlas confirmation candidate

This directory archives the frozen single-seed MFI L4 audit and its executable source.

## Result

At the declared response displacement `0.08`, all frozen gates passed:

| Metric | Result |
|---|---:|
| Concept endpoints | 8 across 4 repeated two-slot codes |
| Pair distances | 28 |
| Transport geometry distortion | `1.4513596e-05` |
| Round-trip geometry distortion | `1.3128981e-07` |
| Random-kernel-perturbed retraction control median distortion | `7.8881497e-05` |
| Transport/random ratio | `0.1839924` |
| Pairwise wins | `28/28` |
| Exact access after transport/round trip | `1.0 / 1.0` |
| Held-out access after transport/round trip | `1.0 / 1.0` |

The audit supports local endpoint-wise response transport between two distinct finite-response level sets in one GPT-2 LoRA parameter chart. The added kernel perturbation in the control was scale-matched to the direct-retraction displacement; the final control displacement was not explicitly norm-matched.

## Layout

- `source/`: runnable source and frozen configuration
- `colab/`: notebook-safe launcher
- `release_assets/`: uploadable experiment ZIP and launcher
- `evidence/v2_1_2/`: raw result archive plus extracted records
- `CLAIM_BOUNDARY.md`: permitted and excluded interpretations
- `STATISTICAL_NOTE.md`: dependence of pairwise comparisons
- `ERRATA.md`: metadata wording correction
- `REPRODUCE.md`: execution instructions
- `SHA256SUMS.txt`: artifact integrity manifest

Scientific status: **prospectively frozen single-seed L4 confirmation candidate**, not multi-seed or cross-model confirmation.
