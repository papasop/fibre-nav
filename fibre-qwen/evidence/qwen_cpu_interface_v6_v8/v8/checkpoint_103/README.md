# V8 checkpoint 103 — audited, paused and resumable

The newly supplied raw archive contains **103/128** checkpoints. Its original
decision is `PAUSED`, `complete=false`, `overall_pass=false`. This supersedes the
earlier 18-point log as the latest available evidence; the earlier snapshot remains
preserved. It is not a completed experimental pass.

## Verified observations

- All seven supplied result ZIPs passed CRC and their complete payload hash checks.
  The six stage archives contain 16, 32, 48, 64, 80 and 96 records; each record is
  byte-identical to the corresponding record in the 103-point main archive.
- Every frozen-source file matches the originally delivered V8 source. Run lock,
  schedule and baseline hashes match. Every recorded summary field reproduces
  under the unchanged evaluator and its pause handler; all 103 log rows match.
- Independent Boolean checks find **103/103** primary choices and bounded outputs
  correct. Each of the four first-pass formats is **16/16**.
- All **24/24** invalid parser inputs are rejected. The **39/39** available repeats
  match their first pass exactly in scores, constrained output/configuration and
  numerical diagnostics.
- Same-forward bit logits are unchanged by the mask in **103/103** records.
  The maximum aligned generation-versus-last-projection logit difference is **0**.
- Six observations fail the retained *legacy* cross-path diagnostic. That metric
  was explicitly diagnostic-only in the frozen V8 protocol; it does not change
  the original failed V6 verdict. No tolerance is changed in this audit.
- Runtime environment matches the frozen V6 baseline. Session duration is
  **5426.95 seconds (90.45 minutes)**, consistent with the 90-minute session budget.
  The handler also catches keyboard interruption, so the generic pause marker
  alone is not definitive proof of the interruption cause.

The incomplete global compiler, numerical and integrity flags remain false by
design until all 128 checkpoints finish. No pointwise answer or aligned numerical
failure was found among the 103 completed records. This is an external compiled
interface with 16 distinct model prompts, not parameter-memory computation or
native language-generalization evidence.

## Resume the unchanged V8

Run the original `COLAB_LAUNCHER_QWEN_COMPILER_CPU_V8.py` and upload both:

1. `qwen_compiler_cpu_v8.zip` (the original source package).
2. `cpu_compiled_interface_results.zip` from this directory (the 103-point archive).

The next point is `record-01-EQUAL-repeat` (point 104). The remaining work is nine
record-format repeats and all sixteen JSON-format repeats. Keep the frozen source,
environment and gates unchanged. Prefer the same Colab runtime: the resume logic
requires an exact recorded environment match. With model files already cached,
roughly 20–30 minutes is a planning estimate based on this run; a fresh model
download adds time. Resume is not automatically executed by publishing this archive.

## Reproduce this audit without model weights

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 fibre-qwen/evidence/qwen_cpu_interface_v6_v8/v8/checkpoint_103/audit_checkpoint.py
```

`AUDIT.json` is derived evidence. Raw ZIPs and the original session log are
byte-preserved. No pretrained model inference was rerun during this audit.
