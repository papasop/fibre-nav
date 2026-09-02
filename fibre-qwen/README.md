# Fibre-Qwen

Auditable development of a research-oriented Qwen assistant using conversation-derived preference bootstraps, frozen evaluation, and targeted rule retrieval.

**Status: v0.0.1-development. This repository does not contain a validated personal model.**

## Current result

Small-data LoRA training completed but did not survive frozen generation-health evaluation. Targeted prompting is currently more promising than weight training, but remains developmental. R21b-r1 restored generation health from 7/10 to 9/10 and improved declared-answer margins on 8/10 reused items, while semantic success remained 5/10. It therefore remains fail-closed. R21b-r2 is the next executable diagnostic: an auditable deterministic router is compared with a non-deployable retrieval oracle to separate routing error from Qwen3-0.6B generation limits. No R21b-r2 result is claimed in this archive.

## Reproduce

Install Python 3.11+, PyTorch, `transformers==4.56.2`, `accelerate==1.10.1`, `peft==0.17.1`, and `safetensors`. Protocol scripts expose their required paths through `--help`. GPU execution is recommended.

## Repository map

- `protocols/`: executable R20–R21 protocols, including the R21b-r2 router/oracle diagnostic; adapters are deliberately excluded.
- `profiles/`: editable research constitution and retrieval cards.
- `evidence/`: exact summaries and per-item records, including negative results.
- `docs/`: evidence ledger, boundaries, and roadmap.

## Privacy

No raw private conversation, credentials, model cache, or trained adapter is included. The 20 bootstrap pairs are short research judgments distilled from prior dialogue and authorized for this development run; they are not independent annotations.

## Citation boundary

Nothing here establishes moving-response-fibre superiority, continual learning, safety, broad personalization, or a new foundation model. Qwen model licensing and attribution remain governed by the upstream model terms.
