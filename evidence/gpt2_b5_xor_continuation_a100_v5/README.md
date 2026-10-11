# GPT-2 B5 V5: Complete XOR Truth-Table Development Coverage

**Recorded outcome:** `B5_XOR_CONTINUATION_DEVELOPMENT_PASS`, 4/4 input cases.
**Evidence date:** 12 October 2026 (Pacific/Fiji).

One frozen GPT-2 backbone carries separate runtime-routed controller and memory
adapters. The frozen controller selects READ, XOR, WRITE and HALT. The external
runtime performs adapter routing, XOR computation and numerical parameter updates.
The controller seed is **82933**, and the memory-chart seed is **82931**.

Each input case starts with an incorrect output cell, so accepting a no-op does
not satisfy the output-flip criterion. Bits below are `[A, B, sentinel, output]`.

| Input | Preparation → accepted final state | Origin | Minimum exact margin | Minimum development-paraphrase margin | Mean anchor KL |
| --- | --- | --- | ---: | ---: | ---: |
| 00 | `[0,0,1,1] → [0,0,1,0]` | Inherited V4 pass, not rerun | 3.654167 | 3.594391 | 0.003496579 |
| 01 | `[0,1,1,0] → [0,1,1,1]` | Inherited V4 pass, not rerun | 5.089798 | 5.392197 | 0.003139988 |
| 10 | `[1,0,1,0] → [1,0,1,1]` | Restore actual V4 preparation step 600; continue to 1200, then write 1080 steps | 12.399918 | 12.671097 | 0.002506764 |
| 11 | `[1,1,1,1] → [1,1,1,0]` | Restart write from saved V4 preparation; write 1080 steps | 3.339340 | 4.552727 | 0.003642533 |

Original endpoint thresholds remain fixed: minimum exact and paraphrase margins
`0.05`, maximum averaged response drift `0.002`, maximum mean anchor KL `0.005`.
All four final writeback drifts are `1.52587890625e-05`. These budgets apply to
declared anchor aggregates; they do not establish per-prompt or global preservation.

## Evidence and verification

- [Original total result ZIP](gpt2_b5_xor_continuation_a100_results.zip): unchanged;
  includes weights/checkpoints, archived source, the failed V4 parent and earlier B3/B4/B5 parents.
- [Raw summary](recorded/SUMMARY.json), [origins and limits](recorded/CONTINUATION_PLAN.json),
  [architecture](recorded/ARCHITECTURE.json), [phase progress](recorded/PHASE_PROGRESS.json).
- `recorded/` and `source/` are byte-identical convenience copies, not edited results.

Run from the repository root, with Python 3 and no GPU/model download:

```bash
python3 evidence/gpt2_b5_xor_continuation_a100_v5/verify_archive.py
python3 evidence/gpt2_b5_xor_truth_table_a100_v4/verify_archive.py
PYTHONDONTWRITEBYTECODE=1 python3 evidence/gpt2_b5_xor_continuation_a100_v5/source/test_dispatch.py
PYTHONDONTWRITEBYTECODE=1 python3 evidence/gpt2_b5_xor_continuation_a100_v5/source/test_continuation.py
```

The read-only verifier checks CRC, 64 member digests, three parent archives,
recorded gates/minima, inherited records, operation traces, unchanged controller
checkpoint and candidate tensor storage present in the saved final states. It
does not execute GPT-2, deserialize tensors or independently reproduce model readouts.
The 12 + 6 runtime tests are software tests, not additional model evidence.

## A100 launcher and resume

Copy the entire [V5 one-cell PY](GPT2_B5_XOR_CONTINUATION_A100_V5.py) into a Colab
A100 cell. To replay this development continuation, supply the
[verified V4 result](../gpt2_b5_xor_truth_table_a100_v4/gpt2_b5_xor_truth_table_a100_results.zip).
To resume V5, use this directory's matching V5 result ZIP. Uploading the completed
V5 result restores a completed run; it is not a fresh replication.

The default mounts Drive, saves true checkpoints every five steps and retains
three verified backups under `GPT2_B5_XOR_CONTINUATION_A100_V5_RECOVERY`.
The session has no time cap, but phase caps remain preparation 1200 total steps
and writeback 1080 steps per attempt.

The published wrapper adds `mkdir(parents=True, exist_ok=True)` before launching
the unchanged embedded source. This fixes the missing V4 dependency-directory
error in a new Colab runtime without changing the source payload, worker identity,
numerical method, thresholds or resume compatibility. [Repair provenance](WRAPPER_REPAIR.json)
and the [original wrapper](archived_launcher/GPT2_B5_XOR_CONTINUATION_A100_V5_ORIGINAL.py)
are retained. The repair was checked locally; no additional A100 model run is claimed.

## Interpretation

This is four-input **development coverage across V4 and V5**, not four new
independent V5 experiments. Read paraphrases were previously inspected and remain
development data even where the archived JSON field is named `heldout`.
The fixed controller action template was taught in training; the result does not
establish general program synthesis or new control-flow generalisation.

Controller-probe drift is exactly zero, consistent with disjoint adapter update
coordinates. This is an implementation/routing check, not evidence of shared-chart
self-modification. V3 causal controls are retained in the parent, not rerun for
every truth-table input. Independent memory seeds, fresh evaluations and expanded
causal controls remain next targets. XOR coverage is not Boolean completeness;
internal numerical updates, lifelong learning and zero forgetting are not claimed.
