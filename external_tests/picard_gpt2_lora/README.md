# GPT-2 LoRA Picard External Test

The default blank-Colab entry point is
`GPT2_LORA_PICARD_R5_ONE_CLICK.ipynb`. It clones `papasop/fibre-nav`, checks
out frozen tag `v1.6.0`, verifies commit
`bad9b71d6dbbc36775eb14400e14af719b58e4c5`, runs the repository verifier,
installs the frozen r5 dependencies, executes the r5 ten-step audit and
downloads a result ZIP.

r5 uses the same five seeds as the r4 frozen confirmation and is therefore a
measurement-resolution audit, not an independent new-seed confirmation.

```bash
open https://colab.research.google.com/github/papasop/fibre-nav/blob/main/external_tests/picard_gpt2_lora/GPT2_LORA_PICARD_R5_ONE_CLICK.ipynb
python external_tests/picard_gpt2_lora/COLAB_ONE_CLICK.py --source-root . --verify-only
python external_tests/picard_gpt2_lora/COLAB_ONE_CLICK.py --source-root . --run
```

The Python launcher is for in-repository use after checkout. GPU execution is
expected for a full run. The verifier checks the archived r4 confirmation, r5
audit, r1-r3 developmental statuses and raw ZIP hashes.
