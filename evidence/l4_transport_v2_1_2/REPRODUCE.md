# Reproduce

## Google Colab

1. Select an A100 runtime.
2. Upload and run `colab/run_moving_fibre_intelligence_l4_v2_1_2.py`.
3. When prompted, upload `release_assets/moving_fibre_intelligence_l4_v2_1_2.zip`.
4. Download the generated `moving_fibre_intelligence_l4_results_v2_1_2.zip`.

The observed reference run took `6400.6` seconds (about 107 minutes) on its recorded environment. Runtime can vary with package installation, model caching, and accelerator allocation.

## Command line

```bash
python -m pip install -r source/moving_fibre_intelligence_l4_v2_1_2/requirements.txt
python -u source/moving_fibre_intelligence_l4_v2_1_2/run_l4_confirmation.py \
  --config source/moving_fibre_intelligence_l4_v2_1_2/config_audit.json \
  --output results
```

Verify artifacts with:

```bash
sha256sum -c SHA256SUMS.txt
```
