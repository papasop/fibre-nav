# KL + response successor 002: declared reliability gates passed

Frozen code/protocol commit: 65f4405798094f305b1b15455da0c27c68ac0ba4.
Real Qwen3-0.6B, pinned revision, CPU FP32, 4 threads, cached model in offline mode.
Three development seeds: 84031, 84047, 84061. Three prospective new seeds:
84301, 84317, 84329, frozen before the successor execution and not used for tuning.

| Cohort | Original program success | Successor program success | Original MFI task | Successor MFI task |
|---|---:|---:|---:|---:|
| Development | 1/3 (archived) | 3/3 | 11/48 (archived) | 33/48 |
| New seeds | 1/3 (paired) | 3/3 | 19/48 (paired) | 33/48 |

All 12 successor WRITE/OVERWRITE operations accepted. All six final programs passed.
All declared reliability gates passed, including retaining the known development
success, no paired new-seed regression, and exact rollback on original failed
operations. The successor has no real failed operation in this sample; its exception
rollback path is additionally covered by engineering tests. All 18 engineering tests
passed. Exit code 0.

External memory scores 33/48 and no memory 24/48 in each cohort. The successor now
matches the external-memory baseline for this restricted preference-format task;
it does not exceed that baseline or prove broad response-generation improvement.
The original development arm in report.json has zero NEW observations/accuracy null:
its comparator is the archived task_preference_001 record, independently reproduced
by the candidate-1 paired development run. Do not interpret that empty entry as 0%
accuracy or an original-model failure.

No acceptance thresholds were relaxed: response <=0.02, KL <=0.01, margin >=1.
The original 24 steps, 0.5 step norm and eight backtracks remain. Extra KL-gradient
and response-correction work changes compute cost. Runtime 655.65 seconds covers
all nine runs and tasks and partly overlaps the candidate-1 trial on the same host;
it is not a controlled speed comparison or evidence of acceleration.

The original core.py, qwen_l1.py and evaluate_tasks.py are unchanged. The controller
adds local inward response correction and KL-aware directions, with finite gates
and transactional rollback still deciding commits. The response correction is not
strictly null-space motion. Snapshot files and SHA256SUMS bind the result to code.

Limits: three new seeds and correlated binary preference probes do not establish
universal reliability, scaled semantic memory, L2-L5, autonomous policy, or superiority
over external memory. This successor has not yet been deployed/retested on HF.
The existing HF app continues to show its original frozen baseline experiment.
