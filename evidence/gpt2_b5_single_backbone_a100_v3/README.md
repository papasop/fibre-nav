# GPT-2 B5 single-backbone integration — A100 V3

**Recorded decision:** `B5_SINGLE_BACKBONE_DEVELOPMENT_PASS` (11 October 2026).

One GPT-2 backbone has separate memory (`default`) and controller adapters. The external runtime switches adapters exclusively. The original B4 memory chart/writer is preserved; a new rank-16 controller LoRA-B (seed 82933) trains for 600 steps. Base weights and both LoRA-A charts remain frozen.

## Recorded evidence

- 8/8 symbolic readiness episodes: two training-excluded task tuples × four XOR inputs.
- Actual normal operation: `0110 → 0111`.
- Actual stored substitution: `1111 → 1110`.
- Actual restored operation: `0110 → 0111`.
- Reset control: `0000 → 0000`; read feedback changes and restoration recovers it.
- Required positive output flips, unselected contents, read margins and response/KL gates pass for the three positive cases.
- The fixed initial-task controller score probe has maximum absolute change 0 across each of the four recorded memory write operations.

Reset has `pass=false` solely because it does not require an output flip. It is accepted as a separately evaluated reset comparison, not as a nontrivial positive write.

## Scope and interpretation

This is **single-backbone, partitioned-adapter B5 development evidence**. Runtime routing, XOR and numerical gradient writing remain external services. Controller-score preservation is consistent with inactive-adapter isolation; it does not demonstrate interference-free self-modification in one shared update chart. The fixed score probe is not a global guarantee over all instructions or prompts.

Memory seed 82931 and B4 states are reused. The new controller seed is not an independent memory replication. Readiness tasks were already inspected in V2, so V3 is development testing. All four XOR inputs are covered symbolically, not by four independent prepared actual-memory cases. No full independent B5 confirmation, arbitrary programming, native attention-to-weight updating or Line C result is established.

## Verification and running

From repository root, with standard Python only:

```bash
python evidence/gpt2_b5_single_backbone_a100_v3/verify_archive.py
python evidence/gpt2_b5_single_backbone_a100_v3/source/test_dispatch.py
```

The archive verifier checks CRC, all 40 hashes, recorded splits, traces, gates, controls and architecture declarations. It does not reload the model or independently rerun the experiment. The 10 self-tests cover routing, fixed-action runtime checks and training/evaluation separation.

For actual A100 execution, run `GPT2_B5_SINGLE_BACKBONE_A100_V3.py` in Colab. Upload a passed B4/V2 result to start, or a matching V3 result to resume. Do not upload the source/launcher ZIP as data. Drive backup and resume are included.

The original result ZIP contains tensor checkpoints, parent B4 archive and archived sources. `recorded/` exposes JSON; `source/` comes directly from the result ZIP. `PROVENANCE.json` and `console_log.txt` preserve identity and supplied run output.
