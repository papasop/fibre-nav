# Reproduction

## Colab

1. Enable an NVIDIA GPU runtime.
2. Upload both files from `release_assets/`.
3. Upload both files to `/content`.
4. Run:

```python
%run /content/run_moving_fibre_intelligence_l3_v1_7_8.py
```

The launcher ignores Jupyter's injected `-f kernel.json` arguments, installs
the frozen requirements, executes seed `81902`, creates a diagnostic ZIP even
on experiment failure, and downloads the final result archive.

Expected output:

```text
/content/moving_fibre_intelligence_l3_results_v1_7_8.zip
```

## Direct execution

```bash
python -m pip install -r src/requirements.txt
python -u src/fibre_memory_audit.py \
  --config src/config_quick.json \
  --output results
```

Compare the generated `summary.json`, `seed_81902.json`, and configuration hash
against `evidence/v1_7_8/` and `SHA256SUMS.txt`.
