# Fibre-Qwen

Auditable development of a research-oriented Qwen assistant using conversation-derived preference bootstraps, frozen evaluation, and targeted rule retrieval.

**Status: v0.0.1-development. This repository does not contain a validated personal model.**

## Current result

Small-data LoRA training completed but did not survive frozen generation-health evaluation. Targeted prompting is currently more promising than weight training, but remains developmental. R21b-r2 achieved exact 10/10 deterministic routing, so retrieval is no longer the leading bottleneck. R21c compared Qwen3-0.6B with Qwen3-8B, but its original lexical evaluator was mechanically invalidated by R21c-r1. Under the transparently revised post-hoc contracts, Qwen3-8B free generation scored 7/10 versus 2/10 for 0.6B. This nominates an 8B free-generation candidate for a future untouched R21d confirmation; it does not validate or deploy a personal model.

## Reproduce

Install Python 3.11+, PyTorch, `transformers==4.56.2`, `accelerate==1.10.1`, `peft==0.17.1`, and `safetensors`. Protocol scripts expose their required paths through `--help`. GPU execution is recommended.

## Repository map

- `protocols/`: executable R20–R21 protocols through the R21c-r1 evaluator audit; adapters are deliberately excluded.
- `profiles/`: editable research constitution and retrieval cards.
- `evidence/`: exact summaries and per-item records, including negative results. R21b-r2 is explicitly marked as transcript-reconstructed because its original result ZIP was unavailable.
- `docs/`: evidence ledger, boundaries, and roadmap.

## Privacy

No raw private conversation, credentials, model cache, or trained adapter is included. The 20 bootstrap pairs are short research judgments distilled from prior dialogue and authorized for this development run; they are not independent annotations.

## Citation boundary

Nothing here establishes moving-response-fibre superiority, continual learning, safety, broad personalization, or a new foundation model. Qwen model licensing and attribution remain governed by the upstream model terms.
