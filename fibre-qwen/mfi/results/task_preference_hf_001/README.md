# First real Hugging Face CPU task reproduction

Space: oahosoh/fibre-qwen-r24a-train (private), free CPU basic, 2 vCPU/16 GB.
Deployment commit: 9774de7796959b945d758e4d731fc63994578947.
Evaluation source commit: 53469b980d66e1874b4cb816f8d13f35d7c361e2.
Triggered manually through the deployed page; no protocol or evaluation source changes.
Cloud execution took 336.32 seconds; completed with scientific-gate exit code 2.

MFI 11/48, external memory 33/48, no memory 24/48. All three aggregate gates failed.
Two of six planned stages accepted; two writes rejected/rolled back and their
subsequent overwrites skipped. Failed/skipped task items remain in the denominator.
This reproduces the earlier local scores, not evidence of a task advantage.

Protocol and evaluated source hashes match the local run exactly. The runtime is
PyTorch 2.8.0+cpu, Transformers 4.56.2, safetensors 0.8.0. See full raw records
and logs under run/. The primary runtime dependencies are pinned; this is not
an assertion that all transitive packages or CPU hardware match the local run.

Downloaded archive SHA256:
ff2f93344f2490127731da7d70b6aa8fbe179b084b6d00b3a464bf0061f6d3c5

Deployment repair: preserve the private iframe query on same-origin status/run/
download requests; suppress HTTP access logs that could record signed query data.
Three fixture-based deployment tests passed before upload. Live status, manual
execution, completion display and result download were subsequently exercised.

No long-context replacement, semantic graph capacity, autonomous self-editing,
multi-tenant isolation, or cost/speed multiplier is established by this experiment.
