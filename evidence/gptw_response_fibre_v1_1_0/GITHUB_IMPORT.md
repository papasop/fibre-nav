# GitHub import instructions

This archive is repository-ready evidence for the GPTW branch of `papasop/neural-fibre-geometry`.

## Suggested destination

Copy this directory to `evidence/gptw_response_fibre_v1_1_0/` without changing the files. Then run:

```bash
cd evidence/gptw_response_fibre_v1_1_0
sha256sum -c SHA256SUMS
python verify_snapshot.py
```

Suggested release tag: `v1.4.0-gptw-natural-text-confirmed`.

## Commit message

```text
Add prospective GPT-2/LoRA natural-text moving-fibre confirmation
```

## Scope of this bundle

This bundle closes the GPT-2 evidence chain used by Sections 3.5 and 5.4 of the manuscript: formal negative, precision repair, new-seed adaptive-value confirmation, codeword current-versus-fixed extension, and prospective natural-English confirmation.

It does **not** contain the manuscript's separately pending ResNet v4.1b, v4.2c/r1 or v4.2d assets, because those exact programs and per-seed records were not available in the assembly workspace. Do not use this bundle to remove those ResNet pending-release statements. Import the corresponding frozen artifacts separately before changing the manuscript's ResNet availability claim.
