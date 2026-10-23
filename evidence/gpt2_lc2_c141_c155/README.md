# LC2 development boundary and numerical diagnostics: C141–C155

Evidence snapshot: 23 October 2026. These selected runs are **development,
confirmation-boundary and diagnostic evidence; LC2 remains open**. C-numbered
runs are not completed validation targets. C35–C37 retain their original local
LC1 development scope and pending independent confirmation.

| Run | Recorded outcome | Evidence included in this import |
| --- | --- | --- |
| C141 | Frozen C140 writer: 42/48 new local seen-class states in each round; raw 32/48, C137 40/48. Six response failures retained. | Original ZIP fully hashed; original manifest/source checked; all96 preparation and288 endpoint records, actual48 preparations and144 proposals, frozen writer/control tensors. |
| C142 | Completed read-only audit retaining `C141_FROZEN_KERNEL_LOCAL_CONFIRMATION_FAILURE`. | Original ZIP fully hashed; compact original audit output/source checked. No new model replay. |
| C148 | Both global continuation and context routing: 79/84 training states in each round; neither qualified. | Original ZIP fully hashed; original records, frozen heads, router, full3211 features, prepared/raw coordinates and shared basis. |
| C150 | Diagonal continuation 79/84; cyclic coupling 80/84 in each round. Four coupled-arm response failures remain. | Same original-archive and actual-tensor tier as C148. Final points are reconstructed from included tensors, rather than duplicated as two large336-point archives. |
| C154 | Completed fixed-direction training diagnostic at two archived positions, 1512 metrics /3025 model forwards reported. Late finite changes disagree with archived linear prediction. | Unchanged delivered source and supplied execution log with a log-derived summary. The293MB original ZIP was not downloaded/rehashed in packaging. |
| C155 | Completed1530 metrics /3062 model forwards reported; all756 FP32 native/functional observation and KL differences are zero. Two fixed-state FP64 diagnostics retained by the original run. | Unchanged delivered source and supplied execution log with a log-derived summary. The381MB original ZIP was not downloaded/rehashed in packaging. |

All84 C148-and-later states are exposed training states, including IDs containing
`confirmation`. C141's historical confirmation role is retained as an original
record, but those states cannot be reused as unseen evaluation. None of these
six runs establishes an excluded-class pass or closes LC2.

## What is checked and what is missing

`ARCHIVE_PROVENANCE.json` distinguishes original bytes actually rehashed during
packaging from log-only records. `MEMBER_INDEX.json` maps each compact member to
its exact original ZIP member and SHA256. Tensor NPZ files are stored once by
content hash under `objects/`; shared raw writer/control/data/basis bytes are not
duplicated across stages. All included arrays retain their original bytes,
shapes, names and dtypes. `RESULT_MANIFEST.json` copies refer to the full original
archives; omitted members do not become available merely because their hashes
are recorded.

No full original result ZIP is included or publicly attached by this import.
Predecessor archives required for a fresh model run remain external. The four
checked ZIPs support compact provenance and arithmetic, not independent model
accuracy, chronology or non-access proof. C154/C155 must stay visibly log-tier
until an actual result supplement is obtained and checked. The supplied CPU
`EXPORT_C154_C155_COMPACT_COLAB.py` can export that supplement from the owner's
Drive without model loading or training. C156 has no execution result and is
not included as evidence.

## Interpretive limits

- C148's two arms both reach79/84; routing superiority is not established.
  Parameters differ36 versus108, and the fixed router is not a certified atlas.
- C150's cyclic arm has216 parameters versus108 in its diagonal comparison;
  its80/84 result is not complete training qualification, a geometric feasibility
  proof or an excluded-input result.
- C154/C155 do not certify AD error or correctness, FP64 ground truth, systematic
  forward-error bounds, convergence, or a causal explanation for failures.
- Acceptance remains response≤.002, mean anchor KL≤.005, read margins≥.05 and
  radius≤20 with the original1e-5 rounding allowance. Failed states stay in the
  original denominator. Rollback/retry or selecting a favourable norm/step cannot
  turn a failed proposal into a pass.

Run from the repository root with Python and NumPy:

```bash
python3 evidence/gpt2_lc2_c141_c155/verify_records.py
```

The report separately counts four original-archive-backed compact records and
two log-only records. Verification completion is not LC2 completion. See
`external_tests/gpt2_lc2_c141_c155/README.md` for the unchanged real CPU protocols.
