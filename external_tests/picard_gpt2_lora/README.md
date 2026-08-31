# GPT-2 LoRA Picard External Test

The default external entry point runs the r5 ten-step resolution audit. r5 uses
the same five seeds as the r4 frozen confirmation and is therefore a
measurement-resolution audit, not an independent new-seed confirmation.

```bash
python external_tests/picard_gpt2_lora/COLAB_ONE_CLICK.py --source-root . --verify-only
python external_tests/picard_gpt2_lora/COLAB_ONE_CLICK.py --source-root . --run
```

GPU execution is expected for a full run. The verifier checks the archived r4
confirmation, r5 audit, r1-r3 developmental statuses and raw ZIP hashes.
