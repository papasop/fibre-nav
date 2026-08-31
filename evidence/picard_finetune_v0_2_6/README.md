# Cached Intrinsic Picard Fine-Tuning v0.2.6

This release freezes and reproduces the v0.2.6 repeated-timing confirmation of a cached intrinsic Picard update against AdamW.

## Confirmed result

On five new evaluation seeds, with no v0.2.6 hyperparameter tuning or target selection:

| Metric | Median result |
| --- | ---: |
| Time-to-equal-loss speedup | 24.46% |
| Fixed-budget speedup | 16.70% |
| Steps-to-target reduction | 9.09% |
| Endpoint loss delta, Picard minus AdamW | -0.000099 |
| Certified response leakage | 8.23e-16 |

All frozen gates passed. The machine-readable status is `PICARD_V0_2_6_REPEATED_TIMING_DUAL_10PCT_SPEEDUP_SUPPORTED`.

## Claim boundary

This is a task-specific result for a frozen-feature ResNet-18/CIFAR-10 classifier trained in the same 20-dimensional float64-certified intrinsic response kernel. It is not an end-to-end fine-tuning result, a GPT-2/LoRA result, a universal optimizer comparison, or a proof of global Picard flow.

## Reproduce in Colab

1. Open a GPU Colab runtime (the frozen run used an NVIDIA A100-SXM4-40GB).
2. Upload and run `colab/picard_finetune_v0_2_6_colab_launcher.py`.
3. When prompted, upload `bundles/picard_finetune_demo_v0_2_6.zip`.
4. The launcher verifies the bundle SHA-256, runs the protocol, prints the frozen gates, and downloads `picard_v0_2_6_results.zip`.

Expected A100 runtime is approximately 12–18 minutes after feature extraction. Dataset and pretrained-weight downloads are excluded from optimizer timing.

## Contents

- `src/`: standalone benchmark source.
- `colab/`: single-process Colab launcher.
- `bundles/`: exact upload bundle accepted by the launcher.
- `evidence/v0_2_6/results/`: summary and all ten per-arm/per-seed records.
- `evidence/v0_2_6/picard_v0_2_6_results.zip`: original returned results archive.
- `PROTOCOL.md`: frozen design, gates, and interpretation.
- `MANIFEST.json` and `SHA256SUMS`: provenance and byte-level integrity.

## Local execution

```bash
python src/picard_finetune_benchmark_v0_2_6.py \
  --device cuda \
  --data-root ./data \
  --outdir ./picard_v0_2_6_results
```

The program exits with code `0` only when every frozen gate passes; otherwise it writes any available evidence and exits fail-closed with code `2`.
