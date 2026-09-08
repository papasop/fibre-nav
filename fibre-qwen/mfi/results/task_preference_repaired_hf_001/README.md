# Repaired writer: Hugging Face reproduction 001

Both manual CPU cohorts completed with exit code 0, in requested order. These are actual HF executions, not the local reference displayed at startup.

Deployment: https://huggingface.co/spaces/oahosoh/fibre-qwen-r24a-train/commit/5a2c91e04ee306bfbc85ff23a39e3921b3daa94f
Writer source: 50d8da2d7ec775af8337c73217b2be28ae0ca711.

| Cohort | Seeds | Operations | MFI | External | No memory | Runner seconds |
|---|---|---|---|---|---|---|
| Existing development | 84031/84047/84061 | 6/6 | 33/48 | 33/48 | 24/48 | 391.54 |
| Previously validated second cohort | 84301/84317/84329 | 6/6 | 33/48 | 33/48 | 24/48 | 365.44 |

All original gates passed: response <= 0.02, KL <= 0.01, margin >= 1.0, all operations accepted, task accuracy above no-memory and at least external-memory. Both cohorts' per-item predictions match the archived local candidate-2 runs. Snapshots match each report's source hashes; scores were recomputed from raw items. The two cohorts use identical code hashes.

Directories retain the downloaded report, protocol, snapshots, per-stage predictions and traces, console log, deployment config, and execution status. `summary.json` records original download ZIP hashes and extrema. `SHA256SUMS` covers extracted files.

Scope: six seeds and twelve WRITE/OVERWRITE operations; one-bit brief/detailed preference, restricted two-token format choice. MFI reads a committed bit and supplies preference text to the model. Four prompts and label reversals are correlated probes. This confirms deployment reproduction, not fresh blind testing, open-ended response quality, full L1-L5 support, autonomous policies, or broad memory capacity. Runner times exclude some startup/download overhead and are not speed benchmarks.
