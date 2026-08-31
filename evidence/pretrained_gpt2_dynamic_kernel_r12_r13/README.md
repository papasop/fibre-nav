# Pretrained GPT-2 dynamic response-kernel evidence: R12-R13

This directory archives a fail-closed development-confirmation chain using
`openai-community/gpt2`, a shared 24-dimensional rank-4 LoRA chart in the last
two `c_attn` modules, SHA-256-bound Tiny Shakespeare data, and deterministic
response measurements with all dropout modules disabled.

## Stages

| Stage | Role | Outcome |
| --- | --- | --- |
| R12a | Engineering preflight | Ineligible: pretrained dropout contaminated fixed-state response measurements |
| R12a-r1 | Determinism repair | Engineering gates passed; no dual advantage at the original wide budgets |
| R12a-r2 | One-seed Pareto development | Current kernel dual-won at four tight budgets; nominated `2e-5` and `5e-5` |
| R12b | Three-new-seed development | Both budgets supported; current dual-won 3/3 seeds at each budget |
| R13 | Five-untouched-seed confirmation | Both frozen budgets supported; current dual-won 5/5 seeds at each budget |

## R13 result

| Global response budget | Median budgeted-AdamW minus current loss | Median source minus current loss | Dual-win seeds |
| ---: | ---: | ---: | ---: |
| `2e-5` | `0.001384894` | `0.001331011` | `5/5` |
| `5e-5` | `0.001366377` | `0.001289566` | `5/5` |

The independent experimental units are the five untouched seeds. The ten
seed-by-budget contrasts are correlated within seed and must not be presented
as ten independent replicates.

## Layout

- `code/`: runnable ZIPs for every stage.
- `launchers/`: standalone Colab launchers for R12a-r2, R12b and R13.
- `protocols/`: frozen machine-readable protocols.
- `results/`: original result archives where available, extracted records, and
  the explicitly documented R12b log recovery.
- `RESULTS.md`: compact result and timing ledger.
- `CLAIM_BOUNDARY.md`: supported and prohibited statements.
- `PROVENANCE.md`: exact archive-status disclosure.
- `verify_snapshot.py`: local fail-closed verification.
- `MANIFEST.sha256`: checksums for all other files.

## Reproduction

Run the desired launcher in Google Colab and upload its matching ZIP from
`code/`. A complete R13 run took approximately 55.7 minutes on an NVIDIA
A100-SXM4-40GB. `--quick` is an environment smoke test only. Exit code `0`
means the declared stage gates passed; exit code `2` is a fail-closed
scientific or numerical outcome.
