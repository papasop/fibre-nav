# GPT-2 Fibre Memory Write-Read Audit v1.3.2

This prospective experiment tests a restricted claim: a random codeword can be
written, read, and rewritten by moving LoRA-B parameters inside a response fibre
while a prospectively frozen anchor response remains within budget.

It does **not** test whether all natural-language memory is fibre position, or
whether AdamW discovers fibre coordinates naturally.

## Colab

Upload the ZIP and `run_fibre_memory_colab_v1_3_2.py`, then run:

```python
%run run_fibre_memory_colab_v1_3_2.py --mode smoke
```

The launcher installs dependencies, runs the frozen JSON configuration, creates
`fibre_memory_results.zip`, and triggers a browser download in Colab.

## Local/GPU

```bash
python -m pip install -r requirements.txt
python fibre_memory_audit.py --config config_quick.json --output results
```

Use `--seeds 81401` for a smoke test. A CUDA GPU is strongly recommended.

## Arms

- `true_current`: memory gradient projected into the current response kernel.
- `random_kernel`: deterministic equal-norm random current-kernel directions;
  the reported control is best-of-N.
- `sign_reversed`: opposite of the selected current-kernel direction.
- `shuffled_label`: writes a prospectively permuted codeword.
- `no_move`: unchanged adapter state.

Every nontrivial proposal is followed by a finite Newton-style response
retraction. Steps violating the global response or anchor-KL budget are rolled
back. Dropout is disabled and all frozen anchor prompts, codewords, token IDs,
configuration, package versions, and per-step diagnostics are serialized.

## Primary gates

For each eligible seed:

1. the true arm must satisfy the finite response and anchor-KL budgets;
2. held-out paraphrased-cue read accuracy must meet the configured gate;
3. true read accuracy must exceed best-of-random by the configured margin;
4. rewriting the same state with the complementary codeword must switch the
   decoded memory while remaining in the same response budget.

Version 1.3.2 evaluates every arm at its last checkpoint satisfying both the
declared response budget and the frozen endpoint KL gate. Raw final diagnostics
and checkpoint truncation are retained. Exact-cue write/read/rewrite gates are
reported as primary; paraphrase generalization gates are reported separately as
secondary. It retains the v1.3.1 separation of write and evaluation targets, scores shuffled
content against the original A codeword, applies the same endpoint eligibility
report to every control arm, and records a source/A/not-A cross-read matrix.
It retains two near-prediction natural-word keys, a two-candidate
contrastive write loss, and response-budget-aware backtracking. Exact-cue
decoding is the primary write/read gate; paraphrase decoding is a secondary
generalization gate. Full-vocabulary anchor KL is a prospectively bounded
endpoint diagnostic, not an undeclared per-step response constraint. The quick
configuration is a development run. Freeze a new seed cohort only
after inspecting its output; do not present quick-run results as confirmation.
