# HF serial overwrite audit 001

Both frozen new seeds completed (exit code 2: scientific task gate failure, not a crash or timeout). Each uses one Qwen3-0.6B chart, one controller, and the same initial response/KL reference throughout twenty alternating WRITE/OVERWRITE operations. No tuning occurred between runs.

| Seed | Accepted operations | MFI format choice | External memory | No memory | Overall |
|---|---|---|---|---|---|
| 84503 | 20/20 | 108/160 | 110/160 | 80/160 | FAIL task-matches-external |
| 84521 | 20/20 | 108/160 | 110/160 | 80/160 | FAIL task-matches-external |

The combined 40/40 operations satisfy original response <= .02, KL <= .01, margin >= 1 gates. Task scores exceed no-memory but fall short of external memory by two items per seed (1.25 percentage points). Do not label this a fully passed agent audit. Raw prediction differences and extrema are in summary.json; scores and source snapshot hashes were independently checked after download.

Per-seed directories retain reports, all 20 stage records, protocol, snapshots, console logs, deployment configuration, and execution status. Original ZIP hashes are in summary.json. The records support small-cohort serial writing endurance under the declared anchors. They do not establish general response preservation, open-ended answer quality, multi-address L2-L5, or autonomous instruction execution. No rollback events occurred in these successful operations; rollback reliability cannot be inferred from these runs alone.

The task is hybrid: read committed MFI bit, inject its preference as text, then use restricted two-token format scoring. Four prompts and label reversals are correlated probes, not 160 independent task samples. Preserve this negative task result when evaluating a future logits-based interface; forced output is an interface control, not evidence of reasoning gains.

HF deployment commit: d9bacd961ae23b80aebb03c004cb6e40d260210b.
Protocol source commit: de8f275b1eb9163976cf7cb98000dbca2c7b3736.
