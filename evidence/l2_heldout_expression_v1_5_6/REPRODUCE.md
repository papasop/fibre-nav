# Reproduce L2

1. Open `colab/run_fibre_memory_colab_l2_v1_5_6.py` in a Colab code cell.
2. Run it on an A100-class runtime and upload
   `release_assets/gpt2_fibre_memory_l2_v1_5_6.zip`.
3. Download `fibre_memory_l2_results_v1_5_6.zip`.
4. Place the untouched ZIP at `evidence/v1_5_6/results.zip`, extract its files
   beside it, regenerate `SHA256SUMS.txt`, and verify the summary protocol and
   configuration hash against `source/.../config_quick.json`.

Notebook `-f` arguments and duplicate upload suffixes are handled by the
launcher. Expected A100 runtime in the original development setup was roughly
25–30 minutes.

