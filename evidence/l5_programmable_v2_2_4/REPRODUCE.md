# Reproduction

Recommended environment: Google Colab with an NVIDIA A100.

1. Run `release_assets/run_moving_fibre_intelligence_l5_v2_2_4.py` in a Colab cell.
2. Select `release_assets/moving_fibre_intelligence_l5_v2_2_4.zip` when prompted.
3. Preserve the generated `moving_fibre_intelligence_l5_results_v2_2_4.zip` unchanged.
4. Compare its SHA-256 and the internal `config_sha256` against this snapshot.

The launcher accepts Colab numeric filename suffixes such as ` (1).zip` and ignores notebook `-f` arguments.

The observed A100 runtime was `872.149826` seconds, excluding environment setup and model download variability.
