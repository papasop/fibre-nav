# R23d provenance

- Protocol: `FIBRE_QWEN_GENERAL_MOVING_RESPONSE_KERNEL_R23D_CONFIRMATORY`
- Mode: untouched training-order-seed confirmation
- Model: `Qwen/Qwen3-0.6B`
- Seeds: `64007, 64013, 64019, 64033, 64037`
- Result ZIP SHA-256: `06ad30ad6d80d7dcdcfc08134a440610a7373065dfd0de97ce5d5277d193aa11`
- Benchmark SHA-256 recorded by launcher: `ca22319a169d088b80d8b1dbceb312c0a138c4063f8a39ecbefc82e86ca28d54`
- Wall time: `1147.875854730606 s`

All five seeds supported the moving-current-kernel arm against both source-frozen projection and response-budgeted LoRA AdamW at both frozen budgets. All numerical and confirmatory gates passed. The seeds vary the training order while holding the base model, authored data, LoRA initialization, chart, budgets and protocol fixed.

The median moving-arm budget utilization was 4.45% and the maximum was 7.12%; therefore both response balls were inactive for the moving arm. This is consistent with strong local response preservation but does not establish a continuous Pareto frontier.
