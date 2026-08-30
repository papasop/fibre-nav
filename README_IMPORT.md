# GitHub ingest: GPT-2/native-LoRA-B Pareto R2 strict audit

This archive is a merge payload for `papasop/neural-fibre-geometry`.

## Import

Copy the archive contents into the repository root while preserving paths. Do
not delete or rename the historical `gpt2_lora_b_v1` evidence. R2 is a new
prospective strict-control protocol and does not revise R1.

Run:

```bash
python verify_gpt2_lora_pareto_r2.py
```

The verifier must print `VERIFIED` before committing. Then review
`README_R2_INSERT.md` and merge its short evidence block into the repository
README at the existing low-response Pareto section. Do not copy the instructions
file verbatim into the public README.

Suggested commit message:

```text
Archive strict GPT-2 LoRA-B Pareto R2 confirmation
```

After pushing, replace the paper's old repository commit with the real pushed
commit. No commit SHA is asserted by this payload.
