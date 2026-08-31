# v1.7.0 - Pretrained GPT-2 dynamic-kernel confirmation

This release imports the R12/R13 pretrained GPT-2 dynamic response-kernel
evidence chain under
`evidence/pretrained_gpt2_dynamic_kernel_r12_r13/`.

## Confirmed result

R13 is a frozen five-untouched-seed confirmation using
`openai-community/gpt2`, SHA-256-bound Tiny Shakespeare data, two adapted final
`c_attn` modules and a shared 24-dimensional rank-4 LoRA chart. Under two
declared global response balls, current-response-kernel identity updates beat
both source-frozen updates and response-budgeted AdamW in 5/5 seeds at each
budget.

Median budgeted-AdamW-minus-current validation-loss gaps:

| Budget | Gap |
| ---: | ---: |
| `2e-5` | `0.001384894` |
| `5e-5` | `0.001366377` |

Median source-minus-current validation-loss gaps:

| Budget | Gap |
| ---: | ---: |
| `2e-5` | `0.001331011` |
| `5e-5` | `0.001289566` |

The five seeds are the independent units; the ten seed-by-budget contrasts are
correlated within seed.

## Development chain

R12a, R12a-r1, R12a-r2 and R12b are preserved as development and repair
stages. The original returned result ZIPs are present for R12a, R12a-r1,
R12a-r2 and R13. R12b is explicitly disclosed as mechanically recovered from a
complete returned Colab log because the original `picard_r12b_results.zip` was
not supplied to the archive build.

## Boundaries

Unconstrained AdamW reached lower validation loss but exceeded the response
budgets by hundreds of times. It is retained as a scope diagnostic, not as a
matched-budget control.

This release does not establish superiority to unconstrained AdamW, ordinary
full-model GPT-2 training, semantic transfer, downstream-task transfer,
inverse-Fisher superiority, lower universal per-step complexity, universal
optimizer superiority or a global Picard theorem.

## Verification

```bash
python evidence/pretrained_gpt2_dynamic_kernel_r12_r13/verify_snapshot.py
```
