# GPTW-V3 zero-upload external reproduction

This entry point reproduces the paper's strongest GPTW result without asking
the tester to upload an author-supplied ZIP. It automatically checks out the
immutable GPTW source release, verifies file identity and archived cohort
semantics, runs the frozen V3 protocol, validates its output, and downloads a
provenance-rich result ZIP.

## One click in Colab

[Open the one-cell notebook in Colab](https://colab.research.google.com/github/papasop/neural-fibre-geometry/blob/v1.5.1-gptw-v3-one-click/external_tests/gptw_v3/GPTW_V3_EXTERNAL_ONE_CLICK.ipynb)

Select an A100 GPU, then choose **Runtime → Run all**. No file upload or manual
parameter entry is required. The program has the protocol's 55-minute hard
limit and downloads `cner_gpt2_lora_natural_prompt_moving_fibre_v3_results.zip`.

## What is pinned

- source tag: `v1.4.0-gptw-natural-text-confirmed`;
- source commit: `236f646c472018a7e38be11fd658519763bc2346`;
- protocol: `CNER_GPT2_LORA_NATURAL_PROMPT_MOVING_FIBRE_V3`;
- original seeds 38171--38178 and frozen natural-English prompt sets;
- all original gates and the 55-minute wall-clock limit.

The wrapper runs both `sha256sum -c SHA256SUMS` and `verify_snapshot.py` before
the GPU experiment. It then requires eight seed summaries, four nodes per seed,
32 total nodes, the frozen decision string, at least 6/8 supporting instances,
and 4/4 passing nodes in every reproduced instance.

## Scientific boundary

This is a **same-cohort external reproduction**. It tests execution and result
reproducibility in an outside Colab session, but does not constitute a new-seed
or new-prompt independent confirmation. Such a confirmation must freeze its
new cohort before observing outcomes and retain the original gates unchanged.

## Command-line preflight

From a local checkout at the pinned commit:

```bash
python external_tests/gptw_v3/COLAB_ONE_CLICK_GPTW_V3.py \
  --source-root . --preflight-only
```
