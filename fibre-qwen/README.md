# Fibre-Qwen

Auditable development of a research-oriented Qwen assistant using conversation-derived preference bootstraps, frozen evaluation, and targeted rule retrieval.

**Status: v0.0.1-development. This repository does not contain a validated personal model.**

## Current result

Small-data LoRA training completed but did not survive frozen generation-health evaluation. Targeted prompting remains more promising than weight training, but is still developmental. R21d generated all 60 records on 20 newly frozen items: generation health was 19/20 for routed 0.6B and 20/20 for both 8B arms. Its lexical evaluator nevertheless assigned 0/20 to every arm and routing covered only 16/20 items. R21d-r1 reproduced the frozen hashes and localized evaluator saturation plus four missing routing triggers per routed model. The automatic model ordering is therefore invalid and human review of 40 blinded pairs remains pending; no personal model has been validated or deployed.

## Reproduce

Install Python 3.11+, PyTorch, `transformers==4.56.2`, `accelerate==1.10.1`, `peft==0.17.1`, and `safetensors`. Protocol scripts expose their required paths through `--help`. GPU execution is recommended.

## Repository map

- `protocols/`: executable R20–R21 protocols through the R21d-r1 frozen-output evaluator audit; adapters are deliberately excluded.
- `profiles/`: editable research constitution and retrieval cards.
- `evidence/`: exact summaries and per-item records, including negative results. R21b-r2 is explicitly marked as transcript-reconstructed because its original result ZIP was unavailable.
- `docs/`: evidence ledger, boundaries, and roadmap.

## Privacy

No raw private conversation, credentials, model cache, or trained adapter is included. The 20 bootstrap pairs are short research judgments distilled from prior dialogue and authorized for this development run; they are not independent annotations.

## Citation boundary

Nothing here establishes moving-response-fibre superiority, continual learning, safety, broad personalization, or a new foundation model. Qwen model licensing and attribution remain governed by the upstream model terms.
