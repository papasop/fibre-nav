# C35–C37 reported development evidence

Read [scope and counts](../../docs/LINE_C_LOCAL_WRITING.md) and
[provenance](PROVENANCE.json) first. `reported/` contains verbatim user-supplied
execution logs and exact summary objects extracted from them. These are not
independently inspected result archives. No GPT-2 result ZIP or weights are
included. Historical B0–B5 archives retain their separate evidence tier.

Run without model downloads or PyTorch:

```bash
python3 evidence/gpt2_line_c_c35_c37/verify_reported_evidence.py
```

This verifies file hashes, summary extraction, counts, stated gates and shared
coefficient identity. It does not verify endpoint measurements, checkpoint
tensors, execution completeness beyond the logs, or statistical generalisation.
