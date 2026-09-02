# Fibre-Qwen

Auditable development of a research-oriented Qwen assistant using conversation-derived preference bootstraps, frozen evaluation, and targeted rule retrieval.

**Status: v0.0.1-development. This repository does not contain a validated personal model.**

## Current result

R23c-r2 connected the first actual moving-response-fibre compute core to Qwen3-0.6B: explicit `DR(theta)`, numerical `ker DR(theta)` projection, current-kernel recomputation, source-frozen control, and finite global response budgets. Its precision audit passed all nine execution gates.

R23d then froze that configuration and evaluated five previously unused training-order seeds. At both frozen budgets, the moving-current-kernel arm achieved lower validation loss than both source-frozen projection and response-budgeted LoRA AdamW in 5/5 seeds; every numerical and confirmatory gate passed. Median loss advantages were 0.07148 and 0.03455 versus source-frozen, and 0.12631 and 0.12380 versus AdamW. This is confirmed only within Qwen3-0.6B, 12 authored learning records, four validation records, six response coordinates, and one restricted 40,960-parameter LoRA chart. The seeds vary training order rather than base-model initialization.

The moving arm used a median 4.45% and maximum 7.12% of the allowed response budget, so the two response balls were not active constraints for that arm. The result does not establish a continuous Pareto frontier, broad capability, personalization, continual learning, safety, deployment readiness, universal optimizer superiority, or a response-fibre theorem.

The earlier R20-R21 personalization and routing experiments remain in the repository as a separate fail-closed development chain; they are not retroactively repaired by R23d.

## Reproduce

Install Python 3.11+, PyTorch, `transformers==4.56.2`, `accelerate==1.10.1`, `peft==0.17.1`, and `safetensors`. Protocol scripts expose their required paths through `--help`. GPU execution is recommended.

## Repository map

- `protocols/`: executable R20–R21 protocols plus the R23c-r2 precision audit and frozen R23d confirmation; adapters are deliberately excluded.
- `profiles/`: editable research constitution and retrieval cards.
- `evidence/`: exact summaries and per-item records, including negative results. R21b-r2 is explicitly marked as transcript-reconstructed because its original result ZIP was unavailable.
- `docs/`: evidence ledger, boundaries, and roadmap.

## Privacy

No raw private conversation, credentials, model cache, or trained adapter is included. The 20 bootstrap pairs are short research judgments distilled from prior dialogue and authorized for this development run; they are not independent annotations.

## Citation boundary

Nothing here establishes moving-response-fibre superiority, continual learning, safety, broad personalization, or a new foundation model. Qwen model licensing and attribution remain governed by the upstream model terms.


Machine-readable status: `R23D_MOVING_RESPONSE_KERNEL_CONFIRMED`.
