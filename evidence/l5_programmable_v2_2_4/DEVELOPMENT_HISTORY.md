# Development history

- **v2.2.0:** exact shared-state operations passed; held-out access did not.
- **v2.2.1:** adaptive training-side worst-item repair; held-out failures remained.
- **v2.2.2:** corrected a causal-read-site confound by preserving the canonical terminal suffix `[address=XX]:`; WRITE and OVERWRITE held-out access passed, with one SWAP item remaining negative.
- **v2.2.3:** adaptive SWAP-horizon polish achieved all-gate development closure in seed `82901`.
- **v2.2.4:** froze the v2.2.3 training configuration, changed to fresh seed `82931` and a fresh held-out expression family, added a `0.05` minimum-margin gate and SWAP-specific controls, and passed all 16 gates.

The archived v2.2.3 result is included for provenance. Earlier failed results should remain available in the broader development record when practical, but they are not confirmation evidence.
