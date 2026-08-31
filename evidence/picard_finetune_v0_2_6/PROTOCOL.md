# Frozen Protocol v0.2.6

Protocol identifier: `GEOMETRIC_INTRINSIC_PICARD_FINETUNE_V0_2_6_REPEATED_TIMING_CONFIRMATORY`

## Frozen configuration

- Model/data: ImageNet-pretrained ResNet-18 frozen features on CIFAR-10.
- Intrinsic chart: fixed 512-to-24 random down-map followed by a float64 response-kernel basis.
- Intrinsic dimension: 20; response rank: 4.
- Optimized state: a 20-by-10 intrinsic linear classification head.
- AdamW learning rate: `0.003`.
- Cached Picard learning rate: `0.12`.
- Strict validation-loss target: `1.6318350233280339`.
- Training steps: 30,000; batch size: 256.
- Evaluation seeds: `22229, 22247, 22259, 22271, 22277`.
- Timing repeats: five identical deterministic trajectories per seed and arm.
- No v0.2.6 tuning or target selection.

The Picard arm precomputes the frozen diagonal inverse pullback metric before timing and reuses it in every update. Synchronized CUDA kernel intervals are measured; feature extraction, downloads, evaluation, setup, and preconditioner construction are excluded.

## Frozen gates

The run passes only if all of the following hold:

- five new evaluation seeds;
- no v0.2.6 tuning or target selection;
- five timing repeats for every seed/arm;
- repeated trajectories and endpoints are numerically identical;
- repeat timing coefficients of variation are at most 10%;
- both arms reach the strict target for every seed;
- median time-to-equal-loss speedup is at least 10%;
- median fixed-budget speedup is at least 10%;
- Picard accuracy is noninferior within 0.5 percentage points for every seed;
- median endpoint-loss delta is at most 0.002 and every delta is at most 0.003;
- median aggregate AdamW target-measurement time is at least two seconds;
- median aggregate full-training measurement time is at least ten seconds;
- float64 response leakage is at most 1e-10.

The original records in `evidence/v0_2_6/results/` are authoritative for the observed values.
