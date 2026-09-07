# Source manifest — v1.5.6

- Frozen development seed: `81801`
- Training mechanism unchanged from v1.5.5
- Comparison gate: worst held-out margin instead of median held-out margin
- Checkpoint rule: weakest training-view accuracy, then weakest margin, then median margin
- A reserve KL: `0.0038`; write/rewrite steps: `360/520`
- Unchanged: prompts, held-out split, controls, response budget, endpoint KL gate
- Scope: L1+L2 only; development, not confirmation

Exact SHA-256 hashes are written into `PACKAGE_SHA256SUMS.txt` when packaged.
