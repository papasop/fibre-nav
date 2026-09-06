# Reproduction instructions

## Google Colab

1. Select an A100 GPU runtime.
2. Upload `release_assets/run_moving_fibre_intelligence_l3_v1_9_6_0.py`.
3. Execute the launcher with `%run`.
4. When prompted, select `release_assets/moving_fibre_intelligence_l3_v1_9_6_0.zip`.
5. Download the generated `moving_fibre_intelligence_l3_results_v1_9_6_0.zip`.

The launcher ignores notebook-injected `-f kernel.json` arguments and accepts browser-renamed copies such as `... (1).zip` after checking their internal structure.

## Local execution

The source archive preserves the relative module layout expected by the bridge. Extract it before running:

```bash
unzip moving_fibre_intelligence_l3_v1_9_6_0.zip -d mfi_v1_9_6_0
python -m pip install -r mfi_v1_9_6_0/moving_fibre_intelligence_l3_v1_9_6_0/requirements.txt
python -u mfi_v1_9_6_0/moving_fibre_intelligence_l3_v1_9_6_0/run_experiment.py \
  --config mfi_v1_9_6_0/moving_fibre_intelligence_l3_v1_9_6_0/config_quick.json \
  --output results_v1_9_6_0
```

GPU execution is strongly recommended. Exact numerical values may vary slightly across hardware and library builds, but the frozen gates must not be altered for a reproduction claim.

## Verification

Verify archive contents against `SHA256SUMS.txt`. Compare the new `summary.json` with `evidence/v1_9_6_0/summary.json` and report every failed gate rather than excluding the run.

