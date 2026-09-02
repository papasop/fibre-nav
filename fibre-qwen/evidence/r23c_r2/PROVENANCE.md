# R23c-r2 provenance

- Protocol: `FIBRE_QWEN_GENERAL_MOVING_RESPONSE_KERNEL_R23C_R2_PRECISION_AUDIT`
- Mode: one-seed general compute-core precision audit
- Model: `Qwen/Qwen3-0.6B`
- Result ZIP SHA-256: `2600a216ba2807f166eed554c2033ce9cb1f9daa7c847a3f15eb78a5fcd62af4`
- Package ZIP SHA-256: `e508937187cbeafdce4fbb0018d91a8075a24d7bd738d1472cd2e5feff14a66c`

R23c-r2 repaired the BF16 response-quantization failure found in R23c-r1 by using FP32 forward computation and FP64 response aggregation. All nine numerical execution gates passed. The observed moving-kernel ordering was developmental and nominated the frozen R23d confirmation; it was not itself a superiority claim.
