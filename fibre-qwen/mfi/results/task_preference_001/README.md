# Frozen preference task smoke 001

Protocol/code frozen before execution at GitHub commit
`ee8bd869c4a352ac26751eeeb3c218bbd06045ff`.
Real Qwen3-0.6B, pinned revision, CPU FP32, four threads; 164.44 seconds.
All 11 engineering tests passed. Scientific aggregate exit code: 2 (failed gates).

| Arm | Correct / planned | Accuracy |
|---|---:|---:|
| MFI workflow | 11 / 48 | 22.92% |
| Ideal external memory | 33 / 48 | 68.75% |
| No memory | 24 / 48 | 50.00% |

Only 2/6 planned MFI stages accepted: seed 84031 WRITE and OVERWRITE.
Seeds 84047 and 84061 rejected WRITE and rolled back; OVERWRITE was skipped.
All failed/skipped items remain in the MFI denominator as workflow failures,
not as observed incorrect model generations. See per-stage status fields.

Within the two accepted stages, MFI and external memory each scored 11/16:
8/8 for detailed preference, 3/8 for brief preference. This descriptive subset
must not replace the planned denominator. The experiment shows neither an
end-to-end MFI gain nor an advantage over external memory. It exposes both
operation reliability failure and imperfect downstream preference following.

Four contexts and two label orders repeat across seeds and stages; 48 is not
48 independent tasks or subjects. No significance claim. No autonomous policy,
free-text response quality, direct semantic parameter access or L2-L5 claim.
No protocol, prompt, seed or budget was changed in response to these scores.

`report.json` separates operation and task gates; each stage JSON contains
operation traces and raw task predictions. Source snapshots and hashes bind
this result to the evaluated code. SHA256SUMS covers this archive's files.
