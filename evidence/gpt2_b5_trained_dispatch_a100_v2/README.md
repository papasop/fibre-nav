# GPT-2 B5 trained model dispatch — A100 V2

Recorded decision: `B5_MODEL_DISPATCH_DEVELOPMENT_PASS`, 11 October 2026.

A separate GPT-2 controller (seed 82932, rank-16 LoRA-B; base/A frozen) is trained for 600 fixed steps on 440 teacher-labelled trajectory contexts from 22 distinct-address XOR tuples. Two complete address tuples are excluded from training. Frozen grammar-constrained token-likelihood action selection passes 8/8 symbolic readiness episodes: two tuples × four XOR inputs.

Actual parameter-memory runs reuse GPT-2 B4 seed 82931 and its archived prepared/substituted states. Each run selects READ, READ, XOR, WRITE and HALT. Normal and restored runs change `[0,1,1,0]` to `[0,1,1,1]`; stored substitution changes `[1,1,1,1]` to `[1,1,1,0]`. Unselected cells and declared endpoint gates pass. Reset changes the read feedback; restoration recovers it; stored-content substitution changes feedback and the computed result.

The reset case's `pass=false` is caused solely by the positive-operation `required_output_flip` check: the reset state computes 0 XOR 0 and remains all zero. The reset control is accepted by its separate comparison checks; it is not counted as a successful nontrivial write.

## Scope

This is a **two-model, single-development-seed learned-dispatch result**. The external runtime computes XOR, performs gradient updates and rejects writes outside the declared task's output address. The controller is not automatically corrected or supplied with the evaluation action sequence. The symbolic readiness check covers all four XOR inputs; actual memory testing covers the recorded prepared/substituted/reset/restored cases, not four independent prepared XOR inputs. Reused development paraphrases are not new independent evaluation data.

No independent multi-seed B5 confirmation, arbitrary instruction programming, one-model self-modification, internal gradient execution, or Line C result is established.

## Files and verification

- Original result ZIP includes controller weights/optimizer, B4 parent, source and traces.
- `recorded/` exposes JSON records without loading tensors.
- `source/` is extracted from the actual result archive, not reconstructed from logs.
- Colab PY/ZIP runs or resumes V2 on A100. Upload a passed B4 result or matching V2 result; do not upload the launcher ZIP as data.
- `console_log.txt` is the supplied run log.
- `PROVENANCE.json` records digests and evidence scope.

From repository root:

```bash
python evidence/gpt2_b5_trained_dispatch_a100_v2/verify_archive.py
python evidence/gpt2_b5_trained_dispatch_a100_v2/source/test_dispatch.py
```

The first command verifies archived records and checksums without training or rerunning GPT-2. It does not certify that the recorded forward passes occurred independently of the supplied archive.
