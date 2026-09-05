# Reproduce v1.9.1 in Colab

Select an A100 runtime. Upload both files from `release_assets/`:

1. `run_moving_fibre_intelligence_l3_v1_9_1.py`
2. `moving_fibre_intelligence_l3_v1_9_1.zip`

Run:

```python
%run /content/run_moving_fibre_intelligence_l3_v1_9_1.py
```

The launcher ignores Jupyter's injected `-f` arguments, installs dependencies,
checks GPT-2 LoRA injection and JSON serialization, forces a fresh extraction,
and downloads:

```text
/content/moving_fibre_intelligence_l3_results_v1_9_1.zip
```

Expected A100 time is approximately 75–100 minutes. Do not change the seed,
holdout list, threshold, gates or controls after seeing the run.
