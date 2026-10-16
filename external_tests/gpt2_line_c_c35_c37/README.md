# C35–C37 reproduction sources

`launchers/` holds the exact delivered single-cell Colab PY files. Each embeds
its own SHA-256-checked source ZIP. C36 uses FIX1, which removes the synthetic
self-test timeout while preserving the scientific worker and protocol.
`source/` is the complete inspectable C37 tree, including inherited C36 FIX1,
C35 and earlier dependencies. Do not flatten or rename inherited modules: source
pins and recursive import paths depend on this structure.

Use an A100 Colab, upload the chosen PY and execute `%run /content/<filename>.py`.
Input must be a compatible **result** ZIP, not a launcher/source ZIP:

| Run | New run parent | Resume |
| --- | --- | --- |
| C35 | Completed compatible C34 result | Matching C35 result |
| C36 FIX1 | Completed passing C35 result | Matching C36 result |
| C37 | Completed passing C36 FIX1 result with pinned freeze identity | Matching C37 result |

The original real-model parent archives and checkpoints are not in this update.
Their absence prevents reproduction from this package alone. This limitation
must remain visible until those artifacts are separately archived. A completed
archive resumed/recounted is not a new evaluation.

For protocol-only source checks, use Python 3:

```bash
python3 external_tests/gpt2_line_c_c35_c37/source/test_c37_protocol.py
```

Torch synthetic tests require a compatible local CPU PyTorch/NumPy environment:

```bash
python3 external_tests/gpt2_line_c_c35_c37/source/test_c37_torch.py
```

These tests are not GPT-2 evaluation. Real runs use CUDA PyTorch already installed
in Colab and isolate the pinned Hugging Face dependencies. The online path
commits one presealed proposal per arm; offline preparation still uses geometry.
