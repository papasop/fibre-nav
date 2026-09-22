# Qwen CPU V9 — external two-step program development

CPU only, Qwen3-1.7B float32, four threads, no training or adapter.
Protocol: `QWEN_CHAIN_CPU_V9`; profile: `two_step_chain`.

## What this tests

For each authored program, the external controller asks the model for
`t = op1(a,b)`, then passes **the actual decoded t** to `op2(t,c)`.
It does not pass an expected answer, repair a wrong intermediate value, or reuse
a cached inference answer. A third call tests `op2(1-t,c)` as a controlled
intervention. The 16 direct single-operation calls serve as within-run controls.

There are 32 programs: every ordered pair of the four operations OR, AND, XOR,
EQUAL appears twice. For operation-pair index i=0..15, the input triples are the
three-bit encodings of `i mod 8` and `(i mod 8) XOR 7`. All eight triples occur,
but this is **32 of the 128 possible operation-pair/input-triple combinations**,
not exhaustive verification. Inventory and order are frozen in FROZEN_PLAN.json.

Total: 16 direct calls + 32 x (step1, step2, flipped-step2) = **112 slots**.
If the first output is not a valid bit, its two dependent slots are recorded as
skipped without model calls; the overall gate fails. Independent programs continue.
For a valid run there are 112 bounded generations, 336 candidate-score forwards,
and 224 diagnostic forwards. Only 16 distinct canonical model prompts occur.
No independent full-program repeat, seed study or new model family is included.

## Frozen gates

All 112 slots must complete; all parsed/compiled fields must match their declared
inputs and actual recorded dependencies. Each original unmasked candidate decision
and bounded result must be correct, with positive margin above the 1e-5 tie floor.
All 32 program endpoints and flipped-input control outputs must be correct; the
observed flip effect must equal the Boolean function's conditional sensitivity.
Insensitive controls are retained and are expected to leave the result unchanged.

Every call must retain exact same-forward bit logits through masking and match
independently evaluated last-position logits (cache off/on) within 1e-5. The old
full-projection-versus-generation gap metric is retained as a diagnostic, exactly
as prospectively specified in V8; it does not retroactively change V6's failure.
The first 16 calls must replay the bundled V6 selected-rule baseline; each of the
96 chain/control calls must agree with its matching fresh direct call in scores,
generation configuration, bounded outputs and numerical records. Recorded runtime
environment must match the baseline, and resume must preserve the run environment.

An observed wrong intermediate remains in the execution trace and can propagate
or be masked by the second Boolean operation. If no first-step error occurs,
the run does not empirically establish handling of naturally occurring errors;
the flipped-input arm tests a deliberate intervention only. Engineering tests
separately inject wrong and malformed outputs.

## Scope and history

This is a prospective development test of externally wired model calls. The
controller implements the program sequence, data transfer and bit-flip operation.
The model supplies only Boolean call outputs. Passing does not establish native
planning, neural parameter memory, autonomous instruction execution, general
language understanding, deployment readiness, or a formal L6 result.

V8's last fully audited raw archive was paused at 103/128; its final pass has not
been confirmed when V9 is frozen. V9 does not replace the remaining V8 audit.
This source is frozen before running V9, but no public preregistration is claimed.

## Colab usage

1. Run the new `QWEN_V9_TWO_STEP.py` in a fresh notebook cell. Do not run the V8 launcher.
2. At the upload prompt, select `qwen_chain_cpu_v9.zip`.
3. For resumption, upload the same source ZIP plus one latest **V9** results ZIP.
   V8 results are not accepted as V9 resume data.

The launcher creates a new session directory, reuses compatible CPU dependencies,
and runs an isolated subprocess without venv/ensurepip or notebook argument leakage.
Google Drive is disabled. Each committed slot refreshes the results ZIP; every
16 slots a stage archive is offered for download. The final result and session log
are offered too. Check browser downloads and keep the final archive.

Default session budget: 90 minutes, including model loading/download inside the
worker. It pauses resumably at a slot boundary. Based on the previous V8 CPU run,
roughly 1–2 hours is a planning estimate for 112 calls; this is not a V9 benchmark.
A cold download or slower host may require another session. Avoid running V8 and
V9 simultaneously on the same CPU runtime.

The launcher prints actual output paths. Main archive:
`.../session_<id>/run/cpu_two_step_chain_results.zip`.

## Local engineering checks (no model weights)

```sh
python3 -m unittest discover -s qwen_chain_cpu_v9 -p 'test_*.py' -v
python3 qwen_chain_cpu_v9/run_cpu.py --out /tmp/v9_check --check-only
```

Tests use archived observations as a mock backend and do not count as a new
pretrained-model experiment. See VALIDATION.md for what was actually checked.
