# PC2 result artifact required

The frozen PC2 source and Colab launcher are present, but the original PC2 result ZIP was not available to the packaging process. Console summary text is deliberately not reconstructed into evidence files.

Run `source/COLAB_LAUNCHER_GPTW_PC2_CPU.py`, preserve the resulting ZIP unchanged, and extract it into this directory. A complete archive must contain:

- `report.json`
- `protocol.json`
- `node_metrics.csv`
- `loo_predictions.csv`
- eight per-seed JSON records
- the complete run log

Expected decision from the prior run summary: `RESPONSE_FIBRE_CAPACITY_OPTIMIZER_ACCESS_NOT_SUPPORTED`. This expectation is not a substitute for the original machine-readable report.
