# V9: audited external two-step interface pass

Original decision: **EXTERNAL_TWO_STEP_INTERFACE_PASS**; `complete=true`,
`overall_pass=true`, **112/112** committed slots. The complete supplied CPU run
took **4468.16 seconds (74.47 minutes)**. No pretrained inference was rerun during
this archival audit.

| Frozen measure | Audited result |
|---|---:|
| Primary candidate choices | 112/112 correct |
| Bounded outputs | 112/112 correct |
| Two-step program endpoints | 32/32 correct |
| Actual-output / flipped-output dependency links | 64/64 matched |
| Fresh direct-call replay | 96/96 exact in scores, bounded records and numerical records |
| Same-forward mask preservation | 112/112 exact |
| Aligned generation versus last-position logits | Maximum absolute difference 0 |
| Minimum primary signed margin | 15.741090774536133 nats |
| Flip intervention changes / invariances | 24 changed, 8 unchanged; all expected |
| Original V6 selected-rule baseline replay | Exact; recorded environment matched |

## What passed

An external deterministic controller obtains `t = op1(a,b)` from frozen
Qwen3-1.7B, passes the actual decoded `t` to `op2(t,c)`, and separately evaluates
`op2(1-t,c)`. The source code, prompts, inventory and gates are unchanged from the
delivered V9 package. The model supplies Boolean call outputs; the controller
provides sequencing, data transfer and intervention. This supports the finite
externally wired interface within the declared test domain.

## Boundaries and retained diagnostics

- There are 32 authored programs, covering 32 of 128 possible ordered operation
  pair / input-triple combinations, and only 16 distinct canonical model prompts.
  Repeated calls are not independent seeds or new semantic tasks.
- No first-step error occurred. Natural error propagation was therefore not
  observed; deliberate bit flips test intervention sensitivity. Mock-engineering
  error injection is separate from these model results.
- Seven calls exceeded the retained legacy cross-path gap diagnostic, with maximum
  difference `1.1444091796875e-5`. The prospectively frozen V9 gate uses aligned
  last-position logits and same-forward masks, which passed. No thresholds were
  revised after observing this run; the original failed V6 verdict is unchanged.
- This is frozen-weight inference with no adapter, parameter-memory write, native
  planning, independent confirmation cohort, or deployment claim. It does not
  establish the Qwen parameter-space address mechanism.
- V8's final 128-point archive has still not been supplied/audited here. This V9
  pass does not promote V8's 103-point paused record to a completed pass.
- The source was frozen before this run; this GitHub import follows the results
  and is not a prospective public preregistration timestamp.

## Payloads and reproducibility

The main raw ZIP, seven stage ZIPs (016 through 112), and original session log are
preserved byte for byte. All stage records match the main archive. Each source
file embedded in every result archive matches the delivered source. All original
stage summaries and the complete final decision reproduce under the frozen
evaluator. Independent code additionally checks Boolean truth, actual dependencies,
flip effects, exact direct replays, log records and raw numerical fields.

`AUDIT.json` is derived; the original ZIPs remain authoritative. `audit_v9.py`
reproduces the audit without downloading or loading model weights:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 fibre-qwen/evidence/qwen_cpu_chain_v9/audit_v9.py
```

The [protocol archive](../../protocols/qwen_chain_cpu_v9/README.md) includes the
original launcher, source ZIP, extracted frozen files and engineering tests.
`MANIFEST.sha256` covers this evidence and the V9 protocol payloads; repository
`provenance/SHA256SUMS` additionally covers that manifest and updated documentation.
