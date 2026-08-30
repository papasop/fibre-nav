# v2 Protocol Completeness Addendum

This addendum supplements the archived v2 protocol in
`../gptw_response_fibre_v1_1_0/experiments/v2_current_vs_fixed/results/protocol.json`.
It does not alter the frozen engine, result JSON files, report, decision or
original SHA-256 manifest.

No numerical result or frozen gate changed. Documentation/protocol-completeness
correction only.

## Frozen Engine Constants

The following constants are declared in the frozen v2 engine
`../gptw_response_fibre_v1_1_0/experiments/v2_current_vs_fixed/run.py`:

| Field | Value |
|---|---|
| Protocol | `CNER_GPT2_LORA_MOVING_FIBRE_CURRENT_VS_FIXED_V2` |
| Model fallback list | `openai-community/gpt2`, `gpt2` |
| Seeds | `26742, 26743, 26744, 26745, 26746, 26747` |
| Layers | `[10, 11]` |
| LoRA rank | `2` |
| LoRA alpha | `4` |
| LoRA trainable tensor | `B` only |
| LoRA fixed tensor initialization | `A ~ Normal(0, 0.02)` |
| Response dimension | `16` |
| Audit radii | `0.20, 0.10, 0.05` |
| Path step | `0.20` |
| Interior nodes | `3` |
| Retraction rounds | `2` |
| Required passing nodes per seed | `2` |
| Required passing seeds | `4` |
| Maximum path-response error | `2e-4` |

TF32 is disabled in the run path, the model is converted to `float64`, and CUDA
is required by the archived certification script.

## Response Definition

For each of four frozen anchors, the response vector records four scalar
features from the next-token logits:

1. target-token logit;
2. mean of the top-3 logits;
3. all-token logit mean;
4. all-token logit standard deviation.

The four anchors therefore produce `4 x 4 = 16` response coordinates.

## Normal-Solve Ridge Formula

The frozen `solve_normal(J, error)` routine computes a minimum-norm correction
through a ridge-stabilized normal solve:

```text
gram  = J J^T
scale = trace(gram) / dim(gram)
ridge = 1e-8 * max(scale, 1e-12)
delta = -J^T (gram + ridge I)^(-1) error
```

This formula determines the finite correction costs used by the v2 current and
source-fixed arms.

## Certification Status

This addendum is descriptive. The certified v2 outcome remains the archived
v1.1.0 result:

- 6/6 seeds passed;
- 18/18 interior nodes passed;
- finest-radius fixed/current correction-cost ratio was 8.60-25.70;
- principal angles were 0.0161-0.0713 rad;
- maximum path-response error was 6.7e-16, below the frozen `2e-4` gate.
