# GPT-2 B5 V4: Historical Four-Input XOR Development Failure

Original outcome: `B5_ACTUAL_XOR_TRUTH_TABLE_DEVELOPMENT_FAIL`, **2/4 passes**.
This negative record is retained unchanged as the parent of V5.

| Case | Recorded outcome |
| --- | --- |
| 00 | Preparation and writeback pass. |
| 01 | Preparation and writeback pass. |
| 10 | Preparation fails: minimum development-paraphrase margin −0.206932; 10/12 correct. |
| 11 | Preparation passes; candidate writeback rejected: exact minimum 0.046349 < 0.05; development-paraphrase minimum −0.149307. |

For 11 the runtime rolls back to the prepared state. A passing gate on that
rolled-back state does not make the rejected candidate a successful write.
V4 deleted the rejected 11 candidate/optimizer; V5 therefore restarts that write
from the saved preparation. The actual 10 preparation checkpoint at step 600 is
available and is resumed in V5. No weights are reconstructed from console logs.

The original [result ZIP](gpt2_b5_xor_truth_table_a100_results.zip), [raw summary](recorded/SUMMARY.json)
and archived `source/` are retained. The 00/01 records are inherited byte-for-byte
in [V5](../gpt2_b5_xor_continuation_a100_v5/), where 10/11 subsequently pass.

```bash
python3 evidence/gpt2_b5_xor_truth_table_a100_v4/verify_archive.py
```

This checks recorded consistency without a GPU; it does not execute the model.
