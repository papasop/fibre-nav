# GPT-2 native-LoRA-B Pareto zero-upload external reproduction

This entry point reproduces the strongest functional audit in the repository
without asking the tester to upload an author-supplied ZIP. It obtains the
repository source, verifies the archived low-response snapshot, imports both
the parent engine and the eight-seed wrapper, and can run either a fast archive
check, an executable smoke test, or the full frozen cohort.

## One click in Colab

[Open the one-cell notebook in Colab](https://colab.research.google.com/github/papasop/neural-fibre-geometry/blob/codex/readme-behaviour-learning-hierarchy/external_tests/gpt2_lora_pareto/GPT2_LORA_PARETO_EXTERNAL_ONE_CLICK.ipynb)

The link above targets the PR branch for review. After release, replace it
with the immutable release tag rather than `main`.

The full run reproduces the original seeds 86841--86848. Scientific role:
same-cohort external reproduction of the prospectively frozen 8-seed/24-node
GPT-2 native-LoRA-B Pareto audit; not an independent new-seed confirmation.

## Local entry points

Verify archived evidence only:

```bash
python external_tests/gpt2_lora_pareto/COLAB_ONE_CLICK_GPT2_LORA_PARETO.py \
  --source-root . --verify-archived
```

Smoke-test the executable stack:

```bash
python external_tests/gpt2_lora_pareto/COLAB_ONE_CLICK_GPT2_LORA_PARETO.py \
  --source-root . --smoke
```

Full frozen same-cohort reproduction:

```bash
python external_tests/gpt2_lora_pareto/COLAB_ONE_CLICK_GPT2_LORA_PARETO.py \
  --source-root . --full
```

The smoke test checks that GPT-2 can be loaded, the parent engine can be
imported, LoRA hooks can be installed and removed, one seed/node can produce a
complete alpha curve, and the output schema matches the frozen protocol shape.

## Frozen expected output

The full run must reproduce:

- attempted seeds: 8;
- supporting seeds: 8;
- noninitial nodes: 24;
- decision: `GPT2_LORA_LOW_RESPONSE_PARETO_CONFIRMED`.

No file upload window should appear.
