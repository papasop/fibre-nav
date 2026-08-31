# GPT-2 LoRA Picard v0.2.6-r5 ten-step resolution audit

This package repeats the frozen v0.2.6-r4 five-seed comparison with target loss checked every 10 rather than every 50 steps. Reusing the same seeds isolates measurement-grid sensitivity; this is not a second independent confirmation. It performs no same-run hyperparameter selection.

- Model: `openai-community/gpt2`
- Adapter: PEFT LoRA rank 4 on `c_attn`
- Data: SHA-256-bound Tiny Shakespeare, fixed 90/10 character split
- Shared warm start: 50 AdamW steps at `5e-4`, excluded from comparison timing
- AdamW: `lr=5e-4`
- Picard: metric exponent `0.5`, `lr=2.0`
- Frozen target validation loss: `3.6593519747257233`
- Target detection: every 10 steps
- Audit gates: median equal-loss speedup >=10% and at least 4/5 positive seed-level speedups

Run through the supplied Colab launcher. A nonzero exit means the frozen scientific gates did not all pass; it is not a launcher failure when `run_summary.json` exists.
