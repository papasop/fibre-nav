# Reproduce

1. Open Google Colab with an A100 runtime.
2. Run `release_assets/run_moving_fibre_intelligence_l1_l5_independent_smoke_v2_3_2_1.py`.
3. Upload `release_assets/moving_fibre_intelligence_l1_l5_independent_smoke_v2_3_2_1.zip` when prompted.
4. Preserve the downloaded result ZIP and compare its summary and hashes.

Expected A100 runtime is approximately 25--45 minutes. Deterministic equality
is environment-sensitive; the scientific criterion is the frozen gate set,
not byte-identical floating-point output.
